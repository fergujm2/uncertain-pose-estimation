#include <rclcpp/publisher.hpp>
#include <rclcpp/rclcpp.hpp>
#include <rclcpp/subscription.hpp>
#include "geometry_msgs/msg/pose_with_covariance_stamped.hpp"
#include "sensor_msgs/msg/camera_info.hpp"
#include <cv_bridge/cv_bridge.hpp>

#include <Eigen/Geometry>

#include <opencv2/aruco/charuco.hpp>
#include <opencv2/aruco/dictionary.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/core/eigen.hpp>

#include "charuco_detector.h"


class PoseEstimatorNode : public rclcpp::Node {
public:
    PoseEstimatorNode();

private:
    void camera_info_callback(sensor_msgs::msg::CameraInfo::ConstSharedPtr msg);

    void image_callback(sensor_msgs::msg::Image::ConstSharedPtr msg);

    std::string aruco_dict_name_;
    double square_size_;
    int squares_x_;
    int squares_y_;

    std::optional<sensor_msgs::msg::CameraInfo> camera_info_;
    std::unique_ptr<ChArUcoDetector> target_detector_;

    std::shared_ptr<rclcpp::Subscription<sensor_msgs::msg::Image>> image_sub_;
    std::shared_ptr<rclcpp::Subscription<sensor_msgs::msg::CameraInfo>> camera_info_sub_;
    std::shared_ptr<rclcpp::Publisher<geometry_msgs::msg::PoseWithCovarianceStamped>> pose_pub_;
};


PoseEstimatorNode::PoseEstimatorNode() 
:
    rclcpp::Node("pose_estimator_node")
{
    aruco_dict_name_ = this->declare_parameter<std::string>("aruco_dictionary", "DICT_6X6_50");
    squares_x_ = this->declare_parameter<int>("squares_x", 5);
    squares_y_ = this->declare_parameter<int>("squares_y", 7);
    square_size_ = this->declare_parameter<double>("square_size", 0.1);


    image_sub_ = create_subscription<sensor_msgs::msg::Image>(
        "/pose_estimator/test_camera/image_rect", 1, 
        std::bind(&PoseEstimatorNode::image_callback, this, std::placeholders::_1));
    
    camera_info_sub_ = create_subscription<sensor_msgs::msg::CameraInfo>(
        "/pose_estimator/test_camera/camera_info", 1, 
        std::bind(&PoseEstimatorNode::camera_info_callback, this, std::placeholders::_1));

    pose_pub_ = create_publisher<geometry_msgs::msg::PoseWithCovarianceStamped>("/pose_estimator/pose", 1);

    cv::namedWindow("pose_estimation", cv::WINDOW_NORMAL);
}


void PoseEstimatorNode::camera_info_callback(sensor_msgs::msg::CameraInfo::ConstSharedPtr msg) {
    // If target detector is  already initialized, ignore msg
    if (target_detector_) {
        return;
    }

    cv::Mat camera_matrix_(cv::Size(3, 3), CV_64F);
    for (size_t row = 0; row < 3; row++) {
        for (size_t col = 0; col < 3; col++) {
            // P is flattened 3x4 so P[row, col] = P[4*row + col]
            camera_matrix_.at<double>(row, col) = msg->p.at(4 * row + col);
        }
    }

    // using rectified image, so no distortion
    cv::Mat distortion_coeffs({0.0, 0.0, 0.0, 0.0, 0.0});
    target_detector_ = std::make_unique<ChArUcoDetector>(
        aruco_dict_name_, 
        squares_x_, 
        squares_y_, 
        square_size_, 
        camera_matrix_, 
        distortion_coeffs);

    RCLCPP_INFO(get_logger(), "Camera info received, ChArUco detector initialized");
}


void PoseEstimatorNode::image_callback(sensor_msgs::msg::Image::ConstSharedPtr msg) {
    // If we don't have a target detector yet, igore msg
    if (!target_detector_) {
        RCLCPP_WARN_THROTTLE(
            get_logger(),
            *get_clock(),
            1000,
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
    cv::Mat annotated;
    auto pose = target_detector_->process(cv_ptr->image, annotated);

    if (pose) {
        target_detector_->drawPose(*pose, annotated);
        cv::imshow("pose_estimation", annotated);
    } else {
        cv::imshow("pose_estimation", cv_ptr->image);
    }

    cv::waitKey(1);


    // TODO publish results
}


int main(int argc, char **argv) {
    rclcpp::init(argc, argv);

    auto node = std::make_shared<PoseEstimatorNode>();
    
    rclcpp::spin(node);

    rclcpp::shutdown();
    return 0;
}