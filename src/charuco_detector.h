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


class ChArUcoDetector {
public:
    ChArUcoDetector(
        const std::string& aruco_dict_name, 
        int squares_x, 
        int squares_y, 
        double square_size, 
        cv::Mat camera_matrix, 
        cv::Mat distortion_coeffs);

    std::optional<Eigen::Isometry3d> process(const cv::Mat& frame, cv::Mat& annotated);

    double boardHeight() const;
    double boardWidth() const;

    void drawPose(const Eigen::Isometry3d& pose, cv::Mat& image) const;

private:
    cv::Ptr<cv::aruco::DetectorParameters> params_ = cv::aruco::DetectorParameters::create();
    cv::Ptr<cv::aruco::CharucoBoard> board_;

    cv::Mat camera_matrix_;
    cv::Mat distortion_coeffs_;
};