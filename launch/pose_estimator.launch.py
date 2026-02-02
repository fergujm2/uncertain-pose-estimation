from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():

    pkg_share = get_package_share_directory('charuco_pose_estimation')

    params_file = os.path.join(
        pkg_share,
        'config',
        'parameters.yaml'
    )

    return LaunchDescription([
        Node(
            package='charuco_pose_estimation',
            executable='pose_estimator',
            name='pose_estimator',
            parameters=[params_file],
            output='screen'
        )
    ])
