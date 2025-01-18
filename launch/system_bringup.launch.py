import os

from launch_ros.actions import Node
from launch import LaunchDescription
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    share_dir = get_package_share_directory('mr_manipulator')

    model_file = os.path.join(share_dir, 'models', 'model.p')

    hand_drawing_node = Node(
        package='mr_manipulator',
        executable='hand_drawing',
        name='hand_drawing_node',
        parameters=[
            {'model_path': model_file}
        ]
    )

    waypoint_transformer_node = Node(
        package='mr_manipulator',
        executable='waypoint_transformer',
        name='waypoint_transformer_node'
    )

    return LaunchDescription([
        hand_drawing_node,
        waypoint_transformer_node
    ])
