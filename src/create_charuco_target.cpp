#include "charuco_board.h"

#include <rclcpp/rclcpp.hpp>
#include <opencv2/highgui.hpp>


int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);

    auto node = std::make_shared<rclcpp::Node>("charuco_board_generator");

    std::string aruco_dict_name = node->declare_parameter<std::string>("aruco_dict_name", "DICT_4X4_1000");
    double square_size = node->declare_parameter<double>("square_size", 0.015);
    int squares_x = node->declare_parameter<int>("squares_x", 17);
    int squares_y = node->declare_parameter<int>("squares_y", 11);
    int dpi = node->declare_parameter<int>("dpi", 300);

    RCLCPP_INFO(node->get_logger(),
        "Generating ChArUco board image with %s: square_size=%.4f, %dx%d at %d dpi",
        aruco_dict_name.c_str(), square_size, squares_x, squares_y, dpi);

    auto board = generate_charuco_board(aruco_dict_name, squares_x, squares_y, square_size);

    cv::Mat image;
    board->draw(
        cv::Size(dpi * squares_x / 2, dpi * squares_y),
        image,
        100,
        1
    );

    cv::imwrite("calibration_target.png", image);

    rclcpp::shutdown();
    return 0;
}

