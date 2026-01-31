#pragma once

#include <opencv2/aruco/charuco.hpp>
#include <opencv2/aruco/dictionary.hpp>
#include <optional>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/base/Matrix.h>


// struct Pose3Gaussian {
//     gtsam::Pose3 mean;
//     gtsam::Matrix6 covariance;
// };


class CharucoPoseEstimator {
public:
    CharucoPoseEstimator(
        const std::string& aruco_dict_name, 
        int squares_x, 
        int squares_y, 
        double square_size, 
        cv::Mat camera_matrix, 
        cv::Mat distortion_coeffs);

    std::optional<gtsam::Pose3> process(
        const cv::Mat& frame,
        cv::Mat& annotated);

    double board_height() const;

    double board_width() const;

private:
    void detect_corners(
        cv::Mat& annotated, 
        std::vector<cv::Point2f>& charuco_corners,
        std::vector<int>& charuco_ids);
    
    std::optional<gtsam::Pose3> estimate_board_pose(
        cv::Mat& annotated,
        std::vector<cv::Point2f>& charuco_corners,
        std::vector<int>& charuco_ids);

    void draw_board_pose(const gtsam::Pose3& pose, cv::Mat& image) const;

    cv::Ptr<cv::aruco::DetectorParameters> params_ = cv::aruco::DetectorParameters::create();
    cv::Ptr<cv::aruco::CharucoBoard> board_;

    cv::Mat camera_matrix_;
    cv::Mat distortion_coeffs_;
};