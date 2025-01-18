import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseArray, Pose
from visualization_msgs.msg import Marker, MarkerArray
from tf2_ros import Buffer, TransformListener
import tf_transformations


class WaypointDisplayNode(Node):
    def __init__(self):
        super().__init__('waypoint_display_node')

        # Initialize TF2 components
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # Subscribe to the PoseArray topic
        self.pose_array_subscription = self.create_subscription(
            PoseArray,
            '/waypoints',
            self.pose_array_callback,
            10
        )

        # Publisher for RViz markers
        self.marker_publisher = self.create_publisher(MarkerArray, '/waypoint_markers', 10)

        self.get_logger().info("3D Waypoint Display Node with RViz markers has been started.")

    def transform_pose(self, pose, target_frame, source_frame):
        """
        Transform a pose from source_frame to target_frame using tf2.
        """
        try:
            transform = self.tf_buffer.lookup_transform(
                target_frame,
                source_frame,
                rclpy.time.Time()
            )

            translation = transform.transform.translation
            rotation = transform.transform.rotation
            
            # Convert pose to a 4x4 transformation matrix
            pose_matrix = tf_transformations.translation_matrix(
                [pose.position.x, pose.position.y, pose.position.z])
            pose_quaternion = [pose.orientation.x, pose.orientation.y,
                               pose.orientation.z, pose.orientation.w]
            pose_matrix = tf_transformations.concatenate_matrices(
                pose_matrix, tf_transformations.quaternion_matrix(pose_quaternion))

            # Convert transform to a 4x4 transformation matrix
            transform_matrix = tf_transformations.translation_matrix(
                [translation.x, translation.y, translation.z])
            transform_quaternion = [rotation.x, rotation.y,
                                    rotation.z, rotation.w]
            transform_matrix = tf_transformations.concatenate_matrices(
                transform_matrix, tf_transformations.quaternion_matrix(transform_quaternion))

            # Apply transformation
            transformed_matrix = tf_transformations.concatenate_matrices(
                transform_matrix, pose_matrix)

            # Extract transformed position and orientation
            position = tf_transformations.translation_from_matrix(transformed_matrix)
            orientation = tf_transformations.quaternion_from_matrix(transformed_matrix)

            # Create new transformed pose
            transformed_pose = Pose()
            transformed_pose.position.x = position[0]
            transformed_pose.position.y = position[1]
            transformed_pose.position.z = position[2]
            transformed_pose.orientation.x = orientation[0]
            transformed_pose.orientation.y = orientation[1]
            transformed_pose.orientation.z = orientation[2]
            transformed_pose.orientation.w = orientation[3]

            return transformed_pose
        except Exception as e:
            self.get_logger().error(f"Failed to transform pose: {e}")
            return None

    def pose_array_callback(self, msg: PoseArray):
        """Callback to handle received PoseArray messages."""
        if not msg.poses:
            self.get_logger().warn("Received an empty PoseArray.")
            return

        transformed_poses = []
        for pose in msg.poses:
            transformed_pose = self.transform_pose(
                pose, "base_link", msg.header.frame_id)
            if transformed_pose:
                transformed_poses.append(transformed_pose)

        if not transformed_poses:
            self.get_logger().warn("No poses successfully transformed.")
            return

        # Create a MarkerArray
        marker_array = MarkerArray()

        for i, pose in enumerate(transformed_poses):
            # Create a sphere marker for each pose
            marker = Marker()
            marker.header.frame_id = "base_link"
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.ns = "waypoints"
            marker.id = i
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD
            marker.pose = pose
            marker.scale.x = 0.1  # Marker size
            marker.scale.y = 0.1
            marker.scale.z = 0.1
            marker.color.r = 1.0  # Red color
            marker.color.g = 0.0
            marker.color.b = 0.0
            marker.color.a = 1.0  # Fully opaque

            # Add marker to the MarkerArray
            marker_array.markers.append(marker)

        # Publish the MarkerArray
        self.marker_publisher.publish(marker_array)
        self.get_logger().info(f"Published {len(marker_array.markers)} markers.")
    

def main(args=None):
    rclpy.init(args=args)

    node = WaypointDisplayNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Node interrupted by user.")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
