#include "charuco_board.h"
#include "charuco_detector.h"

#include <Eigen/Geometry>
#include <opencv2/core/eigen.hpp>


ChArUcoDetector::ChArUcoDetector(
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
    board_ = generateBoard(aruco_dict_name, squares_x, squares_y, square_size);
}

Eigen::Isometry3d openCVToEigen(cv::Mat& R, cv::Vec3d& t) {
    cv::Mat rotation;

    // R can be either 3x3 rotation matrix, or 3x1/1x3 Rodrigues vector
    if (R.rows * R.cols == 3) {
        cv::Rodrigues(R, rotation);
    } else {
        rotation = R;
    }

    Eigen::Isometry3d pose = Eigen::Isometry3d::Identity();
    Eigen::Matrix3d rotation_eigen = pose.rotation();
    cv::cv2eigen(rotation, rotation_eigen);
    pose.linear() = rotation_eigen;
    pose.translation().x() = t[0];
    pose.translation().y() = t[1];
    pose.translation().z() = t[2];

    return pose;
}

std::optional<Eigen::Isometry3d> ChArUcoDetector::process(const cv::Mat& frame, cv::Mat& annotated) {
    frame.copyTo(annotated);

    std::vector<int> marker_ids;
    std::vector<std::vector<cv::Point2f>> marker_corners;

    cv::aruco::detectMarkers(annotated, board_->dictionary, marker_corners, marker_ids, params_);
    if (marker_ids.size() == 0) {
        return {};
    }

    cv::aruco::drawDetectedMarkers(annotated, marker_corners, {});
    std::vector<cv::Point2f> charuco_corners;
    std::vector<int> charuco_ids;
    cv::aruco::interpolateCornersCharuco(
        marker_corners, marker_ids,
        annotated, board_,
        charuco_corners, charuco_ids,
        cv::Mat(), cv::Mat()
    );

    if (charuco_ids.size() == 0) {
        return {};
    }

    cv::Scalar color = cv::Scalar(255, 0, 0);
    cv::aruco::drawDetectedCornersCharuco(annotated, charuco_corners, {}, color);

    cv::Mat rvec;
    cv::Vec3d tvec;
    bool valid = cv::aruco::estimatePoseCharucoBoard(
        charuco_corners,
        charuco_ids,
        board_,
        camera_matrix_,
        distortion_coeffs_,
        rvec,
        tvec
    );

    if (!valid) {
        return {};
    }

    return openCVToEigen(rvec, tvec);
}

double ChArUcoDetector::boardHeight() const {
    return board_->getChessboardSize().height * board_->getSquareLength();
}

double ChArUcoDetector::boardWidth() const {
    return board_->getChessboardSize().width * board_->getSquareLength();
}

void eigenToOpenCV(const Eigen::Isometry3d& pose, cv::Mat& R_out, cv::Vec3d& t_out) {
    Eigen::Matrix3d R = pose.rotation();
    cv::eigen2cv(R, R_out);
    Eigen::Vector3d t = pose.translation();
    t_out = cv::Vec3d(t.x(), t.y(), t.z());
}

void ChArUcoDetector::drawPose(const Eigen::Isometry3d& pose, cv::Mat& image) const {
    cv::Mat R;
    cv::Vec3d t;
    eigenToOpenCV(pose, R, t);
    cv::drawFrameAxes(image, camera_matrix_, distortion_coeffs_, R, t, 0.1f);
}