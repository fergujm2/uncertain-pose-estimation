#include <gtsam/geometry/Pose3.h>
#include <gtsam/geometry/Quaternion.h>
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

    std::string aruco_dict_name_;
    double square_size_;
    int squares_x_;
    int squares_y_;

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

    // using rectified image, so no distortion
    cv::Mat distortion_coeffs({0.0, 0.0, 0.0, 0.0, 0.0});
    pose_estimator_ = std::make_unique<CharucoPoseEstimator>(
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
    if (!pose_estimator_) {
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
    std::optional<gtsam::Pose3> pose = pose_estimator_->process(cv_ptr->image, annotated);

    // Publish annotated image
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

    // Publish pose
    geometry_msgs::msg::PoseWithCovarianceStamped pose_msg;
    pose_msg.header.stamp = msg->header.stamp;
    pose_msg.header.frame_id = msg->header.frame_id;
    
    auto t = (*pose).translation();
    pose_msg.pose.pose.position.x = t.x();
    pose_msg.pose.pose.position.y = t.y();
    pose_msg.pose.pose.position.z = t.z();

    gtsam::Quaternion q = (*pose).rotation().toQuaternion();
    pose_msg.pose.pose.orientation.x = q.x();
    pose_msg.pose.pose.orientation.y = q.y();
    pose_msg.pose.pose.orientation.z = q.z();
    pose_msg.pose.pose.orientation.w = q.w();

    pose_pub_->publish(pose_msg);
}


int main(int argc, char **argv) {
    rclcpp::init(argc, argv);

    auto node = std::make_shared<PoseEstimatorNode>();
    
    rclcpp::spin(node);

    rclcpp::shutdown();
    return 0;
}