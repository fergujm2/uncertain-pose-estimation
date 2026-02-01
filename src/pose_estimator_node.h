#pragma once

#include <optional>

#include <gtsam/geometry/Pose3.h>
#include <gtsam/base/Matrix.h>
#include <gtsam/linear/NoiseModel.h>
#include <gtsam/geometry/Cal3_S2.h>

#include <rclcpp/logging.hpp>
#include <rclcpp/publisher.hpp>
#include <rclcpp/rclcpp.hpp>
#include <rclcpp/subscription.hpp>
#include "geometry_msgs/msg/pose_with_covariance_stamped.hpp"
#include "sensor_msgs/msg/camera_info.hpp"
#include <cv_bridge/cv_bridge.hpp>

#include <opencv2/aruco/charuco.hpp>
#include <opencv2/aruco/dictionary.hpp>


struct Pose3Gaussian {
    gtsam::Pose3 mean;
    gtsam::Matrix6 covariance;
};


class PoseEstimatorNode : public rclcpp::Node {
public:
    PoseEstimatorNode();

private:
    std::optional<Pose3Gaussian> process_frame(
        const cv::Mat& frame,
        cv::Mat& annotated);
    
    Pose3Gaussian optimize_board_pose(
        const std::vector<cv::Point3f>& points_3d,
        const std::vector<cv::Point2f>& points_2d,
        const gtsam::Pose3& pose_init);
    
    void camera_info_callback(sensor_msgs::msg::CameraInfo::ConstSharedPtr msg);

    void image_callback(sensor_msgs::msg::Image::ConstSharedPtr msg);

    void log_results(const Pose3Gaussian& pose, int elapsed_ms) const;

    void draw_board_pose(const gtsam::Pose3& pose, cv::Mat& image) const;

    cv::Ptr<cv::aruco::DetectorParameters> params_ = cv::aruco::DetectorParameters::create();
    cv::Ptr<cv::aruco::CharucoBoard> board_;

    std::optional<cv::Mat> camera_matrix_cv_;
    std::optional<gtsam::Cal3_S2> camera_matrix_gtsam_;
    cv::Mat distortion_coeffs_;

    gtsam::SharedDiagonal pixel_noise_;
    std::string aruco_dict_name_;
    double square_size_;
    int squares_x_;
    int squares_y_;

    std::shared_ptr<rclcpp::Subscription<sensor_msgs::msg::Image>> image_sub_;
    std::shared_ptr<rclcpp::Subscription<sensor_msgs::msg::CameraInfo>> camera_info_sub_;
    std::shared_ptr<rclcpp::Publisher<geometry_msgs::msg::PoseWithCovarianceStamped>> pose_pub_;
    std::shared_ptr<rclcpp::Publisher<sensor_msgs::msg::Image>> image_pub_;
};