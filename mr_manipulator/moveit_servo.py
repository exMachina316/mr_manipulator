#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PointStamped, TwistStamped
from std_msgs.msg import Bool

class HandFollowerServoNode(Node):
    def __init__(self):
        super().__init__('ur5_hand_follower_servo')

        # Predefined z-axis value and velocity scaling factor
        self.predefined_z = 0.2  # Adjust based on your workspace
        self.velocity_scale = 0.6  # Scale velocity for smoother movement

        # Subscribe to the hand position
        self.hand_subscriber = self.create_subscription(
            PointStamped, 
            "/hand_position", 
            self.hand_position_callback, 
            10
        )

        # Publisher for MoveIt Servo commands
        self.servo_cmd_pub = self.create_publisher(TwistStamped, '/servo_server/delta_twist_cmds', 10)

        # Enable Servo (if required by your setup)
        # self.servo_enable_pub = self.create_publisher(Bool, '/servo_server/enable', 10)
        # self.enable_servo()

        self.get_logger().info("UR5 Hand Follower Servo Node Initialized")

    def enable_servo(self):
        enable_msg = Bool()
        enable_msg.data = True
        self.servo_enable_pub.publish(enable_msg)

    def hand_position_callback(self, msg):
        # Calculate the required movement in x, y based on hand position
        target_x = msg.point.x
        target_y = msg.point.y

        # Convert the hand position into velocity commands for MoveIt Servo
        twist_cmd = TwistStamped()
        twist_cmd.header.stamp = self.get_clock().now().to_msg()
        twist_cmd.header.frame_id = 'base_link'

        # Set velocity based on difference in position (scaled for smooth control)
        twist_cmd.twist.linear.x = (target_x - 0.0) * self.velocity_scale  # Adjust x
        twist_cmd.twist.linear.y = (target_y - 0.0) * self.velocity_scale  # Adjust y
        twist_cmd.twist.linear.z = 0.0  # Keep constant or adjust if desired

        # Publish the velocity command
        self.servo_cmd_pub.publish(twist_cmd)
        self.get_logger().info(f"Sending Twist Command: x={twist_cmd.twist.linear.x}, y={twist_cmd.twist.linear.y}")

def main(args=None):
    rclpy.init(args=args)
    node = HandFollowerServoNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
