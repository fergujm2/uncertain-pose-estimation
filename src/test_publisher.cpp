#include <rclcpp/publisher.hpp>
#include <rclcpp/rclcpp.hpp>
#include <rclcpp/subscription.hpp>
#include "sensor_msgs/msg/camera_info.hpp"
#include <cv_bridge/cv_bridge.hpp>
#include <ament_index_cpp/get_package_share_directory.hpp>
#include <filesystem>
#include <string>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/calib3d.hpp>


class TestPublisherNode : public rclcpp::Node {
public:
    TestPublisherNode();

private:
    void timer_callback();
    
    void load_images();

    void init_camera_info();

    rclcpp::TimerBase::SharedPtr timer_;

    sensor_msgs::msg::CameraInfo camera_info_;

    std::vector<cv::Mat> image_raw_;
    std::vector<cv::Mat> image_rect_;
    
    cv::Size image_size_;
    cv::Mat K_;
    cv::Mat D_;
    cv::Mat new_K_;

    std::shared_ptr<rclcpp::Publisher<sensor_msgs::msg::Image>> image_rect_pub_;
    std::shared_ptr<rclcpp::Publisher<sensor_msgs::msg::CameraInfo>> camera_info_pub_;
};


TestPublisherNode::TestPublisherNode() 
:
    rclcpp::Node("pose_estimator_node")
{
    timer_ = this->create_wall_timer(std::chrono::milliseconds(1000),
        std::bind(&TestPublisherNode::timer_callback, this));
    
    camera_info_pub_ = create_publisher<sensor_msgs::msg::CameraInfo>("/pose_estimator/test_camera/camera_info", 1);
    image_rect_pub_ = create_publisher<sensor_msgs::msg::Image>("/pose_estimator/test_camera/image_rect", 1);

    load_images();

    if (image_raw_.size() > 0) {
        init_camera_info();
    }
}


void TestPublisherNode::init_camera_info() {
    // Camera calibration from files in data dir
    K_ = (cv::Mat_<double>(3,3) << 
        494.05852697357193, 0.0, 290.91913816163384,
        0.0, 495.695216059593, 253.6723146547693,
        0.0, 0.0, 1.0
    );

    D_ = (cv::Mat_<double>(1,5) <<
        0.012437475326254561,
        0.05418237601821273,
        0.0022192782212463328,
        -0.004173308623215928,
        -0.2732789551543133
    );

    image_size_ = image_raw_[0].size();
    new_K_ = cv::getOptimalNewCameraMatrix(K_, D_, image_size_, 1.0, image_size_, 0);

    camera_info_.height = image_size_.height;
    camera_info_.width  = image_size_.width;
    camera_info_.distortion_model = "plumb_bob";

    camera_info_.d.resize(5);
    for (int i = 0; i < 5; ++i)
        camera_info_.d[i] = D_.at<double>(0,i);

    // Fill k (3x3) and p (3x4)
    for (int r = 0; r < 3; ++r) {
        for (int c = 0; c < 3; ++c) {
            camera_info_.k[r*3 + c] = new_K_.at<double>(r,c);
            camera_info_.p[r*4 + c] = new_K_.at<double>(r,c);
        }
        camera_info_.p[r*4 + 3] = 0.0;
    }

    // Identity rectification
    for (int i = 0; i < 9; ++i)
        camera_info_.r[i] = (i%4==0) ? 1.0 : 0.0;

    camera_info_.binning_x = 0;
    camera_info_.binning_y = 0;
    camera_info_.roi.x_offset = 0;
    camera_info_.roi.y_offset = 0;
    camera_info_.roi.height = 0;
    camera_info_.roi.width = 0;
    camera_info_.roi.do_rectify = false;
}


void TestPublisherNode::load_images() {
    // Get path to installed images
    std::filesystem::path pkg_share = ament_index_cpp::get_package_share_directory("bayesian_pose_estimation");
    std::filesystem::path images_dir = pkg_share / "test_data" / "images";

    if (!std::filesystem::exists(images_dir) || !std::filesystem::is_directory(images_dir)) {
        RCLCPP_WARN(get_logger(), "Images folder does not exist: %s", images_dir.string().c_str());
        return;
    }

    size_t loaded_count = 0;
    for (const auto &entry : std::filesystem::directory_iterator(images_dir)) {
        cv::Mat img = cv::imread(entry.path().string(), cv::IMREAD_COLOR);
        if (!img.empty()) {
            image_raw_.push_back(img);
            loaded_count++;
        } else {
            RCLCPP_WARN(get_logger(), "Failed to load image: %s", entry.path().string().c_str());
        }
    }

    if (loaded_count == 0) {
        RCLCPP_WARN(get_logger(), "No images loaded from %s", images_dir.string().c_str());
    } else {
        RCLCPP_INFO(get_logger(), "Loaded %zu images from %s", loaded_count, images_dir.string().c_str());
    }
}


void TestPublisherNode::timer_callback() {
    static size_t img_index = 0;

    if (image_raw_.empty()) {
        RCLCPP_WARN(get_logger(), "No images to publish");
        return;
    }

    // Undistort the image
    cv::Mat undistorted;
    cv::undistort(image_raw_[img_index], undistorted, K_, D_, new_K_);

    // Convert cv::Mat to ROS Image
    cv_bridge::CvImage cv_img;
    cv_img.encoding = "bgr8";
    cv_img.image = undistorted;
    cv_img.header.stamp = this->now();

    auto msg = cv_img.toImageMsg();

    // Publish image
    image_rect_pub_->publish(*msg);

    // Publish camera info
    camera_info_.header.stamp = this->now();
    camera_info_pub_->publish(camera_info_);

    RCLCPP_INFO(get_logger(), "Published image %zu / %zu", img_index + 1, image_raw_.size());

    // Move to next image
    img_index = (img_index + 1) % image_raw_.size();
}


int main(int argc, char **argv) {
    rclcpp::init(argc, argv);

    auto node = std::make_shared<TestPublisherNode>();
    
    rclcpp::spin(node);

    rclcpp::shutdown();
    return 0;
}