# ChArUco Pose Estimation with Uncertainty

A ROS2 package for estimating the pose of ChArUco checkerboards as well as the uncertainty of the estimate, given the known 2D pixel projection noise.

OpenCV’s standard checkerboard/ChArUco pose estimation returns only a single pose without any measure of confidence. 
This package additionally computes a full 6×6 pose covariance, allowing downstream systems to reason about uncertainty.

## Running the Demo

1. Install dependencies: `OpenCV` and `GTSAM`
2. `colcon build` this repository
3. Run the test data publisher: `ros2 run charuco_pose_estimation test_data_publisher`
4. Run the pose estimator: `ros2 launch charuco_pose_estimation pose_estimator.launch.py`
5. See the estimated poses: `ros2 run rqt_image_view rqt_image_view -t /pose_estimator/annotated_image`

Additionally, you should see output like the following showing pose mean and covariance:

```bash
[pose_estimator-1] ===== Pose Estimate (2 ms elapsed)=====
[pose_estimator-1] Position (mean, std) [m]:
[pose_estimator-1]     x:   0.09436  ,    0.00090
[pose_estimator-1]     y:  -0.21464  ,    0.00131
[pose_estimator-1]     z:   0.77204  ,    0.00304
[pose_estimator-1] Rotation matrix (mean):
[pose_estimator-1]     [  0.600, -0.792, -0.109 ]
[pose_estimator-1]     [  0.781,  0.551,  0.295 ]
[pose_estimator-1]     [ -0.173, -0.262,  0.949 ]
[pose_estimator-1] Rotation std dev [rad]:
[pose_estimator-1]     roll :  0.017
[pose_estimator-1]     pitch:  0.020
[pose_estimator-1]     yaw  :  0.004
[pose_estimator-1] =========================
```

## IO

### Input msgs 

* Rectified (undistorted) camera image: `sensor_msgs::msg::Image`
* Camera information (for projection matrix `p`): `sensor_msgs::msg::CameraInfo`

### Output msgs

* Annotated image with checkerboard/pose overlay: `sensor_msgs::msg::Image` at `/pose_estimator/annotated_image`

* Estimated pose with covariance: `geometry_msgs::msg::PoseWithCovarianceStamped` at `/pose_estimator/pose`

This is in tangent space Gaussian format where the pose that takes checkerboard coordinates to camera frame coordinates is distributed according to 

```
T = T_mean * expmap(dT),  dT ~ N(0, T_cov).
```

where `T_mean` is the `msg.pose.pose` part (i.e. in SE(3)) part and `T_cov` is the 6x6 covariance matrix `msg.pose.covariance` in row major format. 
This is given in GTSAM's convention where twists have rotation first: `xi = [rot_vec; trans_vec]`.

## Setup

1. Select, download, and print a physical ChArUco checkerboard from [calib.io](https://calib.io/pages/camera-calibration-pattern-generator) 
2. Measure the checkerboard square size on the board.
3. Ensure that you have a calibrated camera currently publishing rectified image data and its projection matrix.
4. Determine a good value for your camera's pixel noise, which accounts for all errors involved with detecting corners and projecting points. A good place to start is the reprojection error output by your intrinsic calibration.
5. Fill out all of the parameters in `config/parameters.yaml`, rebuild and rerun.
