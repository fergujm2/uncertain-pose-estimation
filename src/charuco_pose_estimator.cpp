#include "charuco_pose_estimator.h"
#include "charuco_board.h"

#include <gtsam/geometry/Pose3.h>
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


std::optional<gtsam::Pose3> CharucoPoseEstimator::estimate_board_pose(
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

    gtsam::Pose3 pose = cv_to_gtsam(rvec, tvec);
    draw_board_pose(pose, annotated);

    return pose;
}


std::optional<gtsam::Pose3> CharucoPoseEstimator::process(const cv::Mat& frame, cv::Mat& annotated) {
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