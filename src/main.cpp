#include <gtsam/geometry/Pose3.h>
#include <gtsam/geometry/Quaternion.h>
#include <rclcpp/logging.hpp>
#include <rclcpp/publisher.hpp>
#include <rclcpp/rclcpp.hpp>
#include <rclcpp/subscription.hpp>
#include "geometry_msgs/msg/pose_with_covariance_stamped.hpp"
#include "sensor_msgs/msg/camera_info.hpp"
#include <cv_bridge/cv_bridge.hpp>

#include "charuco_pose_estimator.h"


class PoseEstimatorNode : public rclcpp::Node {
public:
    PoseEstimatorNode();

private:
    void camera_info_callback(sensor_msgs::msg::CameraInfo::ConstSharedPtr msg);

    void image_callback(sensor_msgs::msg::Image::ConstSharedPtr msg);

    void log_results(const Pose3Gaussian& pose, int elapsed_ms) const;

    std::string aruco_dict_name_;
    double square_size_;
    int squares_x_;
    int squares_y_;
    double pixel_noise_sigma_;

    std::optional<sensor_msgs::msg::CameraInfo> camera_info_;
    std::unique_ptr<CharucoPoseEstimator> pose_estimator_;

    std::shared_ptr<rclcpp::Subscription<sensor_msgs::msg::Image>> image_sub_;
    std::shared_ptr<rclcpp::Subscription<sensor_msgs::msg::CameraInfo>> camera_info_sub_;
    std::shared_ptr<rclcpp::Publisher<geometry_msgs::msg::PoseWithCovarianceStamped>> pose_pub_;
    std::shared_ptr<rclcpp::Publisher<sensor_msgs::msg::Image>> image_pub_;
};


PoseEstimatorNode::PoseEstimatorNode() 
:
    rclcpp::Node("pose_estimator_node")
{
    aruco_dict_name_ = this->declare_parameter<std::string>("aruco_dictionary", "DICT_6X6_50");
    squares_x_ = this->declare_parameter<int>("squares_x", 5);
    squares_y_ = this->declare_parameter<int>("squares_y", 7);
    square_size_ = this->declare_parameter<double>("square_size", 0.1);
    pixel_noise_sigma_ = this->declare_parameter<double>("pixel_noise_sigma", 1.0);

    image_sub_ = create_subscription<sensor_msgs::msg::Image>(
        "/pose_estimator/test_camera/image_rect", 1, 
        std::bind(&PoseEstimatorNode::image_callback, this, std::placeholders::_1));
    
    camera_info_sub_ = create_subscription<sensor_msgs::msg::CameraInfo>(
        "/pose_estimator/test_camera/camera_info", 1, 
        std::bind(&PoseEstimatorNode::camera_info_callback, this, std::placeholders::_1));

    pose_pub_ = create_publisher<geometry_msgs::msg::PoseWithCovarianceStamped>("/pose_estimator/pose", 1);
    image_pub_ = create_publisher<sensor_msgs::msg::Image>("/pose_estimator/annotated_image", 1);
}


void PoseEstimatorNode::camera_info_callback(sensor_msgs::msg::CameraInfo::ConstSharedPtr msg) {
    // If target detector is  already initialized, ignore msg
    if (pose_estimator_) {
        return;
    }

    cv::Mat camera_matrix_(cv::Size(3, 3), CV_64F);
    for (size_t row = 0; row < 3; row++) {
        for (size_t col = 0; col < 3; col++) {
            // P is flattened 3x4 so P[row, col] = P[4*row + col]
            camera_matrix_.at<double>(row, col) = msg->p.at(4 * row + col);
        }
    }

    pose_estimator_ = std::make_unique<CharucoPoseEstimator>(
        aruco_dict_name_, 
        squares_x_, 
        squares_y_, 
        square_size_, 
        camera_matrix_,
        pixel_noise_sigma_);

    RCLCPP_INFO(get_logger(), "Camera info received, ChArUco detector initialized");
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
    // If we don't have a target detector yet, igore msg
    if (!pose_estimator_) {
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
    std::optional<Pose3Gaussian> pose = pose_estimator_->process(cv_ptr->image, annotated);
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
    
    auto t = (*pose).mean.translation();
    pose_msg.pose.pose.position.x = t.x();
    pose_msg.pose.pose.position.y = t.y();
    pose_msg.pose.pose.position.z = t.z();

    gtsam::Quaternion q = (*pose).mean.rotation().toQuaternion();
    pose_msg.pose.pose.orientation.x = q.x();
    pose_msg.pose.pose.orientation.y = q.y();
    pose_msg.pose.pose.orientation.z = q.z();
    pose_msg.pose.pose.orientation.w = q.w();

    for (int r = 0; r < 6; ++r)
        for (int c = 0; c < 6; ++c)
            pose_msg.pose.covariance[r*6 + c] = (*pose).covariance(r, c);
    
    pose_pub_->publish(pose_msg);
}


int main(int argc, char **argv) {
    rclcpp::init(argc, argv);

    auto node = std::make_shared<PoseEstimatorNode>();
    
    rclcpp::spin(node);

    rclcpp::shutdown();
    return 0;
}