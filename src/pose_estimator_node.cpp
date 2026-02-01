#include "pose_estimator_node.h"

#include "charuco_board.h"
#include "corner_factor.h"

#include <gtsam/geometry/Cal3_S2.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/Symbol.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <gtsam/nonlinear/Marginals.h>

#include <opencv2/core/eigen.hpp>
#include <opencv2/highgui.hpp>


PoseEstimatorNode::PoseEstimatorNode() 
:
    rclcpp::Node("pose_estimator"), 
    distortion_coeffs_({0.0, 0.0, 0.0, 0.0, 0.0})
{
    // Init ChArUco board from params
    std::string aruco_dict_name = this->declare_parameter<std::string>("aruco_dictionary");
    int squares_x = this->declare_parameter<int>("squares_x");
    int squares_y = this->declare_parameter<int>("squares_y");
    double square_size = this->declare_parameter<double>("square_size");
    board_ = generate_charuco_board(aruco_dict_name, squares_x, squares_y, square_size);

    // Save the board to file for debugging, etc
    cv::Mat image;
    int pixels_per_square = 300;
    cv::Size size(pixels_per_square * squares_x, pixels_per_square * squares_y);
    board_->draw(size, image, 100, 1);

    std::ostringstream ss;
    ss << "target_board_"
        << aruco_dict_name << "_"
        << squares_x << "x" << squares_y << ".png";

    std::string filename = ss.str();
    cv::imwrite(filename, image);
    RCLCPP_INFO(get_logger(), 
        "Saved target ChArUco board image to file: %s\n"
        "Ensure that your physical board matches this image!", 
        filename.c_str());

    // Init pixel noise model from params
    double pixel_noise_sigma = this->declare_parameter<double>("pixel_noise_sigma");
    pixel_noise_ = gtsam::noiseModel::Isotropic::Sigma(2, pixel_noise_sigma);

    // Subscribers from topic name params
    std::string image_rect_topic = this->declare_parameter<std::string>("image_rect_topic");
    image_sub_ = create_subscription<sensor_msgs::msg::Image>(image_rect_topic, 1, 
        std::bind(&PoseEstimatorNode::image_callback, this, std::placeholders::_1));
    
    std::string camera_info_topic = this->declare_parameter<std::string>("camera_info_topic");
    camera_info_sub_ = create_subscription<sensor_msgs::msg::CameraInfo>(camera_info_topic, 1, 
        std::bind(&PoseEstimatorNode::camera_info_callback, this, std::placeholders::_1));
    
    // Publishers
    pose_pub_ = create_publisher<geometry_msgs::msg::PoseWithCovarianceStamped>("/pose_estimator/pose", 1);
    image_pub_ = create_publisher<sensor_msgs::msg::Image>("/pose_estimator/annotated_image", 1);

}


gtsam::Pose3 cv_to_gtsam(cv::Vec3d& rvec, cv::Vec3d& tvec) {
    cv::Mat R;
    cv::Rodrigues(rvec, R);
    gtsam::Matrix3 rot_mat;
    cv::cv2eigen(R, rot_mat);
    
    gtsam::Pose3 pose = gtsam::Pose3(
        gtsam::Rot3(rot_mat), 
        gtsam::Point3(tvec[0], tvec[1], tvec[2]));

    return pose;
}


Pose3Gaussian PoseEstimatorNode::optimize_board_pose(
    const std::vector<cv::Point3f>& board_points,
    const std::vector<cv::Point2f>& image_points,
    const gtsam::Pose3& pose_init)
{
    using namespace gtsam;
    
    // Build factor graph for optimization
    NonlinearFactorGraph graph;
    Key pose_key = gtsam::Symbol('T', 0);

    // Each pair of 2D/3D points constitutes a factor, i.e. an error to be minimized
    for (int i = 0; i < image_points.size(); i++) {
        Point3 point3 = Point3(board_points[i].x, board_points[i].y, board_points[i].z);
        Point2 point2 = Point2(image_points[i].x, image_points[i].y);

        graph.add(CornerFactor(pose_key, point3, point2, *camera_matrix_gtsam_, pixel_noise_));
    }

    // Init values object, needs initial guess for pose
    Values values;
    values.insert(pose_key, pose_init);

    // Do the optimization
    LevenbergMarquardtOptimizer opimizer(graph, values);
    values = opimizer.optimize();
    Marginals marginals(graph, values);

    // Extract solution from values/marginals
    Pose3Gaussian result;
    result.mean = values.at<Pose3>(pose_key);
    result.covariance = marginals.marginalCovariance(pose_key);

    return result;
}


std::optional<Pose3Gaussian> PoseEstimatorNode::process_frame(const cv::Mat& frame, cv::Mat& annotated) {
    frame.copyTo(annotated);

    // First detect the aruco markers within the charuco board
    std::vector<int> marker_ids;
    std::vector<std::vector<cv::Point2f>> marker_corners;
    cv::aruco::detectMarkers(annotated, board_->dictionary, marker_corners, marker_ids, params_);
    cv::aruco::drawDetectedMarkers(annotated, marker_corners, {});

    if (marker_ids.size() == 0) {
        RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 1000,
            "No ArUco markers detected in the image"); 
        return {};
    }
        
    // Then get the charuco corners, i.e. the 2D detected points for pose estimation
    std::vector<cv::Point2f> points_2d;
    std::vector<int> charuco_ids;
    cv::aruco::interpolateCornersCharuco(
        marker_corners, 
        marker_ids,
        annotated, 
        board_,
        points_2d,
        charuco_ids,
        cv::Mat(), 
        cv::Mat()
    );

    cv::Scalar color = cv::Scalar(255, 0, 0);
    cv::aruco::drawDetectedCornersCharuco(annotated, points_2d, {}, color);

    // Pose estimation requires at least 4 corners detected on the board
    int num_corners = points_2d.size();
    if (num_corners < 4) {
        RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 1000,
            "Not enough corners detered in the image: %d < 4", num_corners); 
        return {};
    }

    // Extract 3D points in board frame from charuco corner ids
    std::vector<cv::Point3f> points_3d;
    points_3d.reserve(charuco_ids.size());
    for (size_t i = 0; i < charuco_ids.size(); ++i)
        points_3d.push_back(board_->chessboardCorners[charuco_ids[i]]);

    // Use EPnP method for fast linear solve, good initialization for optimization
    cv::Vec3d rvec;
    cv::Vec3d tvec;
    bool valid = cv::solvePnP(
        points_3d,
        points_2d,
        *camera_matrix_cv_,
        distortion_coeffs_,
        rvec,
        tvec,
        false,
        cv::SOLVEPNP_EPNP   // fast linear
    );

    // Return if solution wasnt valid
    if (!valid) {
        RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 1000,
            "Invalid initial solvePnP solution");  
        return {};
    }

    // Now use gtsam optimizer to locally refine and get uncertainty
    gtsam::Pose3 pose_init = cv_to_gtsam(rvec, tvec);
    Pose3Gaussian pose = optimize_board_pose(points_3d, points_2d, pose_init);
    draw_board_pose(pose.mean, annotated);

    return pose;
}


void gtsam_to_cv(const gtsam::Pose3& pose, cv::Mat& R_out, cv::Vec3d& t_out) {
    cv::eigen2cv(pose.rotation().matrix(), R_out);
    gtsam::Point3 t = pose.translation();
    t_out = cv::Vec3d(t.x(), t.y(), t.z());
}


void PoseEstimatorNode::draw_board_pose(const gtsam::Pose3& pose, cv::Mat& image) const {
    cv::Mat R;
    cv::Vec3d t;
    gtsam_to_cv(pose, R, t);
    cv::drawFrameAxes(image, *camera_matrix_cv_, distortion_coeffs_, R, t, 0.1f);
}


void PoseEstimatorNode::camera_info_callback(sensor_msgs::msg::CameraInfo::ConstSharedPtr msg) {
    // If we already have camera info, ignore msg
    if (camera_matrix_cv_) {
        return;
    }

    camera_matrix_cv_ = cv::Mat(cv::Size(3, 3), CV_64F);

    for (size_t row = 0; row < 3; row++) {
        for (size_t col = 0; col < 3; col++) {
            // P is flattened 3x4 so P[row, col] = P[4*row + col]
            camera_matrix_cv_->at<double>(row, col) = msg->p.at(4 * row + col);
        }
    }

    camera_matrix_gtsam_ = gtsam::Cal3_S2(
        camera_matrix_cv_->at<double>(0,0), // fx
        camera_matrix_cv_->at<double>(1,1), // fy
        camera_matrix_cv_->at<double>(0,1), // skew
        camera_matrix_cv_->at<double>(0,2), // cx
        camera_matrix_cv_->at<double>(1,2)  // cy
    );

    RCLCPP_INFO(get_logger(), "Camera info received, ready to estimate ChArUco board poses");
}


void PoseEstimatorNode::log_results(const Pose3Gaussian& pose, int elapsed_ms) const {

    const auto& t = pose.mean.translation();
    const auto& R = pose.mean.rotation().matrix();  // 3x3 rotation matrix

    double rx_std = std::sqrt(pose.covariance(0,0));
    double ry_std = std::sqrt(pose.covariance(1,1));
    double rz_std = std::sqrt(pose.covariance(2,2));

    double tx_std = std::sqrt(pose.covariance(3,3));
    double ty_std = std::sqrt(pose.covariance(4,4));
    double tz_std = std::sqrt(pose.covariance(5,5));

    RCLCPP_INFO_THROTTLE(
        get_logger(),
        *get_clock(),
        500.0,
        "\n===== Pose Estimate (%d ms elapsed)=====\n"
        "Position (mean, std) [m]:\n"
        "    x: %9.5f  ,  %9.5f\n"
        "    y: %9.5f  ,  %9.5f\n"
        "    z: %9.5f  ,  %9.5f\n"
        "Rotation matrix (mean):\n"
        "    [ %6.3f, %6.3f, %6.3f ]\n"
        "    [ %6.3f, %6.3f, %6.3f ]\n"
        "    [ %6.3f, %6.3f, %6.3f ]\n"
        "Rotation std dev [rad]:\n"
        "    roll : %6.3f\n"
        "    pitch: %6.3f\n"
        "    yaw  : %6.3f\n"
        "=========================",
        elapsed_ms,
        t.x(), tx_std,
        t.y(), ty_std,
        t.z(), tz_std,
        R(0,0), R(0,1), R(0,2),
        R(1,0), R(1,1), R(1,2),
        R(2,0), R(2,1), R(2,2),
        rx_std, ry_std, rz_std
    );
}


void PoseEstimatorNode::image_callback(sensor_msgs::msg::Image::ConstSharedPtr msg) {
    // If we don't have camera info yet, igore msg
    if (!camera_matrix_cv_) {
        RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 1000,
            "Waiting for camera_info before processing images");
        return;
    }

    // Convert using CV bridge
    cv_bridge::CvImageConstPtr cv_ptr;
    try {
        cv_ptr = cv_bridge::toCvShare(msg, "bgr8");
    } catch (const cv_bridge::Exception& e) {
        RCLCPP_ERROR(get_logger(), "cv_bridge error: %s", e.what());
        return;
    }

    // Do the pose estimation
    auto start = std::chrono::high_resolution_clock::now();
    cv::Mat annotated;
    std::optional<Pose3Gaussian> pose = process_frame(cv_ptr->image, annotated);
    auto end = std::chrono::high_resolution_clock::now();
    int elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

    // Publish annotated image no matter what
    cv_bridge::CvImage out;
    out.header = msg->header;
    out.encoding = "bgr8";
    out.image = annotated;

    image_pub_->publish(*out.toImageMsg());

    // Dont publish if the pose is bad
    if (!pose) {
        RCLCPP_WARN_THROTTLE(
            get_logger(),
            *get_clock(),
            1000,   // ms
            "Charuco pose estimation failed"
        );
        return;
    }

    // Log/publish pose
    log_results(*pose, elapsed_ms);

    geometry_msgs::msg::PoseWithCovarianceStamped pose_msg;
    pose_msg.header.stamp = msg->header.stamp;
    pose_msg.header.frame_id = msg->header.frame_id;
    
    auto t = pose->mean.translation();
    pose_msg.pose.pose.position.x = t.x();
    pose_msg.pose.pose.position.y = t.y();
    pose_msg.pose.pose.position.z = t.z();

    gtsam::Quaternion q = pose->mean.rotation().toQuaternion();
    pose_msg.pose.pose.orientation.x = q.x();
    pose_msg.pose.pose.orientation.y = q.y();
    pose_msg.pose.pose.orientation.z = q.z();
    pose_msg.pose.pose.orientation.w = q.w();

    for (int r = 0; r < 6; ++r)
        for (int c = 0; c < 6; ++c)
            pose_msg.pose.covariance[r*6 + c] = pose->covariance(r, c);
    
    pose_pub_->publish(pose_msg);
}