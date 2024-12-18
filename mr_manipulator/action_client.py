import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node
from geometry_msgs.msg import Pose
from ur_interfaces.action import ExecuteWayPoints  # Import your custom action type
import numpy as np

class ExecuteWayPointsClient(Node):
    def __init__(self):
        super().__init__('execute_waypoints_client')

        # Initialize Action Client
        self._action_client = ActionClient(self, ExecuteWayPoints, 'plan_execute_cartesian_path')

    def send_goal(self):
        # Parse waypoints into a PoseArray
        goal_msg = ExecuteWayPoints.Goal()
    
        for i in np.arange(10,21):
            pose = Pose()
            pose.position.x = 0.2
            pose.position.y = 0.1
            pose.position.z = round(0.1 + 0.02*i, 3)
            pose.orientation.x = 0.0
            pose.orientation.y = 0.0
            pose.orientation.z = 0.0
            pose.orientation.w = 1.0  # Identity quaternion
            print(f'Pose {i}: {pose.position.x}, {pose.position.y}, {pose.position.z}')
            goal_msg.waypoints.poses.append(pose)
        
        for i in np.arange(10,21):
            pose = Pose()
            pose.position.x = 0.2
            pose.position.y = round(0.1 + 0.02*i, 3)
            pose.position.z = 0.5
            pose.orientation.x = 0.0
            pose.orientation.y = 0.0
            pose.orientation.z = 0.0
            pose.orientation.w = 1.0  # Identity quaternion
            print(f'Pose {i}: {pose.position.x}, {pose.position.y}, {pose.position.z}')
            goal_msg.waypoints.poses.append(pose)

        for i in np.arange(10,21):
            pose = Pose()
            pose.position.x = 0.2
            pose.position.y = 0.5
            pose.position.z = round(0.7 - 0.02*i, 3)
            pose.orientation.x = 0.0
            pose.orientation.y = 0.0
            pose.orientation.z = 0.0
            pose.orientation.w = 1.0  # Identity quaternion
            print(f'Pose {i}: {pose.position.x}, {pose.position.y}, {pose.position.z}')
            goal_msg.waypoints.poses.append(pose)
        
        for i in np.arange(10,21):
            pose = Pose()
            pose.position.x = 0.2
            pose.position.y = round(0.7 - 0.02*i, 3)
            pose.position.z = 0.3
            pose.orientation.x = 0.0
            pose.orientation.y = 0.0
            pose.orientation.z = 0.0
            pose.orientation.w = 1.0  # Identity quaternion
            print(f'Pose {i}: {pose.position.x}, {pose.position.y}, {pose.position.z}')
            goal_msg.waypoints.poses.append(pose)

        self.get_logger().info(f"Sending goal with {len(goal_msg.waypoints.poses)} waypoints...")

        # Send goal to the action server
        self._action_client.wait_for_server()
        self._send_goal_future = self._action_client.send_goal_async(
            goal_msg,
            feedback_callback=self.feedback_callback
        )
        self._send_goal_future.add_done_callback(self.goal_response_callback)

    def goal_response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().error('Goal rejected by the server')
            return

        self.get_logger().info('Goal accepted, waiting for result...')
        self._get_result_future = goal_handle.get_result_async()
        self._get_result_future.add_done_callback(self.result_callback)

    def feedback_callback(self, feedback_msg):
        feedback = feedback_msg.feedback
        self.get_logger().info(f"Feedback: {feedback.status} | Progress: {feedback.progress * 100:.2f}%")

    def result_callback(self, future):
        result = future.result().result
        if result.success:
            self.get_logger().info(f"Goal succeeded! Message: {result.message}")
        else:
            self.get_logger().error(f"Goal failed: {result.message}")
        rclpy.shutdown()


def main(args=None):
    rclpy.init(args=args)

    client = ExecuteWayPointsClient()

    # Send goal
    client.send_goal()

    rclpy.spin(client)


if __name__ == '__main__':
    main()
