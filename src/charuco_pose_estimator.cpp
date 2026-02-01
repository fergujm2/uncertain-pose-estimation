#include "charuco_pose_estimator.h"
#include "charuco_board.h"
#include "corner_factor.h"

#include <gtsam/geometry/Pose3.h>
#include <gtsam/linear/NoiseModel.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/Symbol.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <gtsam/nonlinear/Marginals.h>

#include <opencv2/core/eigen.hpp>


CharucoPoseEstimator::CharucoPoseEstimator(
    const std::string& aruco_dict_name, 
    int squares_x,
    int squares_y,
    double square_size, 
    cv::Mat camera_matrix,
    double pixel_noise_sigma)
:
    pixel_noise_sigma_(pixel_noise_sigma),
    camera_matrix_(camera_matrix),
    distortion_coeffs_({0.0, 0.0, 0.0, 0.0, 0.0}),
    board_(generate_charuco_board(aruco_dict_name, squares_x, squares_y, square_size))
{}


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


void CharucoPoseEstimator::detect_corners(
    cv::Mat& annotated, 
    std::vector<cv::Point2f>& charuco_corners,
    std::vector<int>& charuco_ids) 
{
    // First detect the aruco markers within the charuco board
    std::vector<int> marker_ids;
    std::vector<std::vector<cv::Point2f>> marker_corners;
    cv::aruco::detectMarkers(annotated, board_->dictionary, marker_corners, marker_ids, params_);
    if (marker_ids.size() == 0) 
        return;

    cv::aruco::drawDetectedMarkers(annotated, marker_corners, {});
    
    // Then get the charuco corners 
    cv::aruco::interpolateCornersCharuco(
        marker_corners, 
        marker_ids,
        annotated, 
        board_,
        charuco_corners,
        charuco_ids,
        cv::Mat(), 
        cv::Mat()
    );

    if (charuco_ids.size() == 0)
        return;

    cv::Scalar color = cv::Scalar(255, 0, 0);
    cv::aruco::drawDetectedCornersCharuco(annotated, charuco_corners, {}, color);
}


Pose3Gaussian CharucoPoseEstimator::optimize_pose(
    const std::vector<cv::Point3f>& board_points,
    const std::vector<cv::Point2f>& image_points,
    const gtsam::Pose3& pose_init)
{
    using namespace gtsam;
    
    NonlinearFactorGraph graph;
    Key pose_key = gtsam::Symbol('T', 0);

    Cal3_S2 intrinsics(
        camera_matrix_.at<double>(0,0), // fx
        camera_matrix_.at<double>(1,1), // fy
        camera_matrix_.at<double>(0,1), // skew
        camera_matrix_.at<double>(0,2), // cx
        camera_matrix_.at<double>(1,2)  // cy
    );
    
    SharedDiagonal pixel_noise = noiseModel::Isotropic::Sigma(2, pixel_noise_sigma_);

    for (int i = 0; i < image_points.size(); i++) {
        Point3 point3 = Point3(board_points[i].x, board_points[i].y, board_points[i].z);
        Point2 point2 = Point2(image_points[i].x, image_points[i].y);

        graph.add(CornerFactor(pose_key, point3, point2, intrinsics, pixel_noise));
    }

    Values values;
    values.insert(pose_key, pose_init);

    LevenbergMarquardtOptimizer opimizer(graph, values);
    values = opimizer.optimize();
    Marginals marginals(graph, values);

    Pose3Gaussian result;
    result.mean = values.at<Pose3>(pose_key);
    result.covariance = marginals.marginalCovariance(pose_key);

    return result;
}


std::optional<Pose3Gaussian> CharucoPoseEstimator::estimate_board_pose(
    cv::Mat& annotated,
    std::vector<cv::Point2f>& points_2d,
    std::vector<cv::Point3f>& points_3d) 
{
    // Use EPnP method for fast linear solve, good initialization for gtsam
    // Note that we assume zero distortion throughout, i.e. using rectified images
    cv::Vec3d rvec;
    cv::Vec3d tvec;
    bool valid = cv::solvePnP(
        points_3d,
        points_2d,
        camera_matrix_,
        distortion_coeffs_,
        rvec,
        tvec,
        false,
        cv::SOLVEPNP_EPNP   // fast linear
    );

    // Return if solution wasnt valid
    if (!valid) return {};

    // Now use gtsam optimizer to locally refine and get uncertainty
    gtsam::Pose3 pose_init = cv_to_gtsam(rvec, tvec);
    Pose3Gaussian pose = optimize_pose(points_3d, points_2d, pose_init);

    // Draw onto annotated image before return
    draw_board_pose(pose.mean, annotated);

    return pose;
}


std::optional<Pose3Gaussian> CharucoPoseEstimator::process(const cv::Mat& frame, cv::Mat& annotated) {
    frame.copyTo(annotated);

    std::vector<cv::Point2f> points_2d;
    std::vector<int> charuco_ids;
    detect_corners(annotated, points_2d, charuco_ids);

    // PnP only works with 4 or more points
    if (points_2d.size() < 4)
        return {};

    // Extract 3D points from charuco corner ids
    std::vector<cv::Point3f> points_3d;
    points_3d.reserve(charuco_ids.size());
    for (size_t i = 0; i < charuco_ids.size(); ++i)
        points_3d.push_back(board_->chessboardCorners[charuco_ids[i]]);

    return estimate_board_pose(annotated, points_2d, points_3d);
}


void gtsam_to_cv(const gtsam::Pose3& pose, cv::Mat& R_out, cv::Vec3d& t_out) {
    cv::eigen2cv(pose.rotation().matrix(), R_out);
    gtsam::Point3 t = pose.translation();
    t_out = cv::Vec3d(t.x(), t.y(), t.z());
}


void CharucoPoseEstimator::draw_board_pose(const gtsam::Pose3& pose, cv::Mat& image) const {
    cv::Mat R;
    cv::Vec3d t;
    gtsam_to_cv(pose, R, t);
    cv::drawFrameAxes(image, camera_matrix_, distortion_coeffs_, R, t, 0.1f);
}