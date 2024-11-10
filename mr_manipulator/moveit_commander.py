#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PointStamped, Pose
from moveit2 import MoveIt2
from moveit2.robots import ur5_robot  # Replace with your actual robot package if different

class HandFollowerNode(Node):
    def __init__(self):
        super().__init__('ur5_hand_follower')

        # Initialize MoveIt2
        self.moveit2 = MoveIt2(
            node=self,
            robot_model=ur5_robot.UR5e(),  # Replace with your UR5 model or move group as needed
            joint_names=['shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint', 
                         'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint']
        )
        self.moveit2.set_max_velocity_scaling_factor(0.5)
        self.moveit2.set_max_acceleration_scaling_factor(0.5)

        # Predefine a z-axis value
        self.predefined_z = 0.2  # Adjust based on your workspace and requirements

        # Subscribe to the hand coordinates topic
        self.hand_subscriber = self.create_subscription(
            PointStamped, 
            "/hand_position", 
            self.hand_position_callback, 
            10
        )

        self.get_logger().info("UR5 Hand Follower Node Initialized")

    def hand_position_callback(self, msg):
        # Extract the x and y coordinates of the hand position
        x = msg.point.x
        y = msg.point.y

        # Get the current pose of the end effector
        current_pose = self.moveit2.get_current_pose()

        # Update the target pose
        target_pose = Pose()
        target_pose.position.x = x
        target_pose.position.y = y
        target_pose.position.z = self.predefined_z
        target_pose.orientation = current_pose.orientation  # Maintain current orientation

        # Set the target pose
        self.moveit2.set_pose_target(target_pose)

        # Plan and move the arm
        success = self.moveit2.go(wait=True)

        # Provide feedback
        if success:
            self.get_logger().info(f"Moved to target hand position: ({x:.2f}, {y:.2f}, {self.predefined_z:.2f})")
        else:
            self.get_logger().warn("Failed to move to target position")

def main(args=None):
    rclpy.init(args=args)
    node = HandFollowerNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
