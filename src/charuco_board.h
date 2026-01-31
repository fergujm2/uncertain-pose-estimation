#pragma once

#include <opencv2/aruco/charuco.hpp>
#include <opencv2/aruco/dictionary.hpp>
#include <string>


static const std::unordered_map<std::string, int> aruco_dict_string_map = {
    {"DICT_4X4_50", cv::aruco::DICT_4X4_50},
    {"DICT_4X4_100", cv::aruco::DICT_4X4_100},
    {"DICT_4X4_250", cv::aruco::DICT_4X4_250},
    {"DICT_4X4_1000", cv::aruco::DICT_4X4_1000},
    {"DICT_5X5_50", cv::aruco::DICT_5X5_50},
    {"DICT_5X5_100", cv::aruco::DICT_5X5_100},
    {"DICT_5X5_250", cv::aruco::DICT_5X5_250},
    {"DICT_5X5_1000", cv::aruco::DICT_5X5_1000},
    {"DICT_6X6_50", cv::aruco::DICT_6X6_50},
    {"DICT_6X6_1000", cv::aruco::DICT_6X6_1000},
    {"DICT_7X7_1000", cv::aruco::DICT_7X7_1000},
    {"DICT_ARUCO_ORIGINAL", cv::aruco::DICT_ARUCO_ORIGINAL},
    {"DICT_APRILTAG_36h11", cv::aruco::DICT_APRILTAG_36h11}
};


inline int aruco_enum_from_string(const std::string& name) {
    auto it = aruco_dict_string_map.find(name);
    if (it == aruco_dict_string_map.end()) {
        std::stringstream ss;
        ss << "Invalid aruco_dictionary '" << name
           << "'. Valid options are:\n";
        for (const auto &kv : aruco_dict_string_map) ss << "  " << kv.first << "\n";
        throw std::runtime_error(ss.str());
    }
    return it->second;
}


inline cv::Ptr<cv::aruco::CharucoBoard> generate_charuco_board(
    const std::string& aruco_dict_name, 
    int squares_x, 
    int squares_y,
    double square_size) 
{
    int aruco_enum = aruco_enum_from_string(aruco_dict_name);
    cv::Ptr<cv::aruco::Dictionary> aruco_dict = cv::aruco::getPredefinedDictionary(aruco_enum);

    double aruco_size = 0.7 * square_size;
    cv::Ptr<cv::aruco::CharucoBoard> board = cv::aruco::CharucoBoard::create(
        squares_x, 
        squares_y, 
        square_size, 
        aruco_size, 
        aruco_dict
    );

    return board;
}