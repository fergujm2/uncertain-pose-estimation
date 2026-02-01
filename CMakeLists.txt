cmake_minimum_required(VERSION 3.8)
project(probabilistic_pose_estimation)

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
target_link_libraries(external_libs INTERFACE ${OpenCV_LIBS} gtsam)

# Charuco tracker executable
add_executable(pose_estimator
  src/main.cpp
  src/pose_estimator_node.cpp
  src/corner_factor.cpp
)

target_compile_features(pose_estimator PUBLIC c_std_99 cxx_std_17)
target_include_directories(pose_estimator PRIVATE src)
target_link_libraries(pose_estimator external_libs)

ament_target_dependencies(pose_estimator
    rclcpp
    geometry_msgs
    sensor_msgs
    cv_bridge
    visualization_msgs
)

install(
  TARGETS pose_estimator
  DESTINATION lib/${PROJECT_NAME}
)

# Test image/camera info publisher
add_executable(test_data_publisher
  src/test_data_publisher.cpp
)

target_compile_features(test_data_publisher PUBLIC c_std_99 cxx_std_17)
target_include_directories(test_data_publisher PRIVATE src)
target_link_libraries(test_data_publisher external_libs)

ament_target_dependencies(test_data_publisher
    rclcpp
    sensor_msgs
    cv_bridge
    ament_index_cpp
)

install(
  TARGETS test_data_publisher
  DESTINATION lib/${PROJECT_NAME}
)

# Other directory installs
install(
  DIRECTORY test_data
  DESTINATION share/${PROJECT_NAME}
)

install(DIRECTORY config
  DESTINATION share/${PROJECT_NAME}
)

install(DIRECTORY launch
  DESTINATION share/${PROJECT_NAME}
)

ament_package()