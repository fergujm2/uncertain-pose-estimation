#include "charuco_pose_estimator.h"
#include "Eigen/src/Geometry/Transform.h"
#include "charuco_board.h"

#include <Eigen/Geometry>
#include <opencv2/core/eigen.hpp>


CharucoPoseEstimator::CharucoPoseEstimator(
    const std::string& aruco_dict_name, 
    int squares_x,
    int squares_y,
    double square_size, 
    cv::Mat camera_matrix, 
    cv::Mat distortion_coeffs)
:
    camera_matrix_(camera_matrix),
    distortion_coeffs_(distortion_coeffs)
{
    board_ = generate_charuco_board(aruco_dict_name, squares_x, squares_y, square_size);
}


Eigen::Isometry3d cv_to_eigen(cv::Vec3d& rvec, cv::Vec3d& tvec) {
    cv::Mat R;
    cv::Rodrigues(rvec, R);

    Eigen::Isometry3d pose = Eigen::Isometry3d::Identity();
    Eigen::Matrix3d rotation_eigen = pose.rotation();
    cv::cv2eigen(R, rotation_eigen);
    pose.linear() = rotation_eigen;
    pose.translation().x() = tvec[0];
    pose.translation().y() = tvec[1];
    pose.translation().z() = tvec[2];

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


std::optional<Eigen::Isometry3d> CharucoPoseEstimator::estimate_board_pose(
    cv::Mat& annotated,
    std::vector<cv::Point2f>& image_points,
    std::vector<int>& charuco_ids) 
{
    // Extract 3D points from charuco corner ids
    std::vector<cv::Point3f> board_points;
    board_points.reserve(charuco_ids.size());
    for (size_t i = 0; i < charuco_ids.size(); ++i)
        board_points.push_back(board_->chessboardCorners[charuco_ids[i]]);

    // Use EPnP method for fast linear solve, good initialization for gtsam
    cv::Vec3d rvec;
    cv::Vec3d tvec;
    bool valid = cv::solvePnP(
        board_points,
        image_points,
        camera_matrix_,
        distortion_coeffs_,
        rvec,
        tvec,
        false,
        cv::SOLVEPNP_EPNP   // fast linear
    );

    if (!valid) {
        return {};
    }

    Eigen::Isometry3d pose = cv_to_eigen(rvec, tvec);
    draw_board_pose(pose, annotated);

    return pose;
}


std::optional<Eigen::Isometry3d> CharucoPoseEstimator::process(const cv::Mat& frame, cv::Mat& annotated) {
    frame.copyTo(annotated);

    std::vector<cv::Point2f> charuco_corners;
    std::vector<int> charuco_ids;
    detect_corners(annotated, charuco_corners, charuco_ids);

    // PnP only works with 4 or more points
    if (charuco_corners.size() < 4)
        return {};

    return estimate_board_pose(annotated, charuco_corners, charuco_ids);
}


double CharucoPoseEstimator::board_height() const {
    return board_->getChessboardSize().height * board_->getSquareLength();
}


double CharucoPoseEstimator::board_width() const {
    return board_->getChessboardSize().width * board_->getSquareLength();
}


void eigen_to_cv(const Eigen::Isometry3d& pose, cv::Mat& R_out, cv::Vec3d& t_out) {
    Eigen::Matrix3d R = pose.rotation();
    cv::eigen2cv(R, R_out);
    Eigen::Vector3d t = pose.translation();
    t_out = cv::Vec3d(t.x(), t.y(), t.z());
}


void CharucoPoseEstimator::draw_board_pose(const Eigen::Isometry3d& pose, cv::Mat& image) const {
    cv::Mat R;
    cv::Vec3d t;
    eigen_to_cv(pose, R, t);
    cv::drawFrameAxes(image, camera_matrix_, distortion_coeffs_, R, t, 0.1f);
}