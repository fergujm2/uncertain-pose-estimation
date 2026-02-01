from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():

    pkg_share = get_package_share_directory('bayesian_pose_estimation')

    params_file = os.path.join(
        pkg_share,
        'config',
        'parameters.yaml'
    )

    return LaunchDescription([
        Node(
            package='bayesian_pose_estimation',
            executable='charuco_tracker',
            name='charuco_tracker',
            parameters=[params_file],
            output='screen'
        )
    ])
