cmake_minimum_required(VERSION 3.8)
project(bayesian_pose_estimation)

find_package(ament_cmake REQUIRED)
find_package(ament_index_cpp REQUIRED)
find_package(rclcpp REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(cv_bridge REQUIRED)

find_package(Eigen3 REQUIRED)
find_package(GTSAM REQUIRED)
find_package(OpenCV REQUIRED)

# Interface library for executables
add_library(external_libs INTERFACE)
target_include_directories(external_libs INTERFACE ${GTSAM_INCLUDE_DIR} ${EIGEN3_INCLUDE_DIR})
target_link_libraries(external_libs INTERFACE ${OpenCV_LIBS} ${GTSAM_LIBRARIES})

# Charuco board image creation executable
add_executable(create_charuco_target
  src/create_charuco_target.cpp
)

target_include_directories(create_charuco_target PRIVATE src)
target_compile_features(create_charuco_target PUBLIC c_std_99 cxx_std_17)
target_link_libraries(create_charuco_target external_libs)

ament_target_dependencies(create_charuco_target
    rclcpp
)

install(
  TARGETS create_charuco_target
  DESTINATION lib/${PROJECT_NAME}
)

# Charuco tracker executable
add_executable(charuco_tracker
  src/main.cpp
  src/charuco_pose_estimator.cpp
)

target_compile_features(charuco_tracker PUBLIC c_std_99 cxx_std_17)
target_include_directories(charuco_tracker PRIVATE src)
target_link_libraries(charuco_tracker external_libs)

ament_target_dependencies(charuco_tracker
    rclcpp
    geometry_msgs
    sensor_msgs
    cv_bridge
    visualization_msgs
)

install(
  TARGETS charuco_tracker
  DESTINATION lib/${PROJECT_NAME}
)

# Test image/camera info publisher
add_executable(test_publisher
  src/test_publisher.cpp
)

target_compile_features(test_publisher PUBLIC c_std_99 cxx_std_17)
target_include_directories(test_publisher PRIVATE src)
target_link_libraries(test_publisher external_libs)

ament_target_dependencies(test_publisher
    rclcpp
    sensor_msgs
    cv_bridge
    ament_index_cpp
)

install(
  TARGETS test_publisher
  DESTINATION lib/${PROJECT_NAME}
)

install(
  DIRECTORY test_data
  DESTINATION share/${PROJECT_NAME}
)

ament_package()