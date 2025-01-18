import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Pose, PoseArray
from visualization_msgs.msg import Marker, MarkerArray
import cv2
import numpy as np

class SkeletonPosePublisher(Node):
    def __init__(self):
        super().__init__('skeleton_pose_publisher')

        # Create a publisher for the PoseArray topic
        self.pose_array_publisher = self.create_publisher(
            PoseArray,
            '/skeleton_pose_array',
            10
        )

        self.get_logger().info("Skeleton Pose Publisher node has been started.")

        # Static file path for the image
        self.image_path = '/root/ur_ws/src/mr_manipulator/mr_manipulator/robot_arm.png'

        # Load the static image
        self.image = cv2.imread(self.image_path)

        # Process the image and publish the PoseArray and MarkerArray
        self.timer = self.create_timer(0.1, self.process_image)

    def skeletonize(self, image):
        # Convert the image to HSV color space
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # Define red color range in HSV
        lower_red = (170, 140, 50)
        upper_red = (180, 255, 255)

        # Create a mask for red regions
        red_mask = cv2.inRange(hsv_image, lower_red, upper_red)

        # Apply Gaussian blur to refine the mask
        blurred_red_mask = cv2.GaussianBlur(red_mask, (15, 15), 0)

        # Apply binary thresholding
        _, binary_thresh = cv2.threshold(
            blurred_red_mask, 1, 255, cv2.THRESH_BINARY)

        # Perform skeletonization using OpenCV ximgproc
        skeleton = cv2.ximgproc.thinning(
            binary_thresh, thinningType=cv2.ximgproc.THINNING_ZHANGSUEN)

        # Extract coordinates of the skeleton
        skeleton_coordinates = np.column_stack(np.where(skeleton > 0))

        # Convert to a list of tuples
        skeleton_coordinates_list = [tuple(coord)
                                      for coord in skeleton_coordinates]

        return skeleton_coordinates_list

    def process_image(self):
        
        image = self.image
        if image is None:
            self.get_logger().error(f"Failed to load image from path: {self.image_path}")
            return

        # Perform skeletonization
        skeleton_coordinates = self.skeletonize(image)

        if not skeleton_coordinates:
            self.get_logger().warn("No skeleton detected.")
            return

        # Create a PoseArray message
        pose_array = PoseArray()
        pose_array.header.stamp = self.get_clock().now().to_msg()
        pose_array.header.frame_id = 'camera_optical'

        for (y, x) in skeleton_coordinates:
            # Populate PoseArray
            pose = Pose()
            pose.position.x = (float(x) - image.shape[1]/2) * 0.01
            pose.position.y = (float(y) - image.shape[0]/2) * 0.01
            pose.position.z = 0.0
            pose.orientation.x = 0.0
            pose.orientation.y = 0.0
            pose.orientation.z = 0.0
            pose.orientation.w = 1.0
            pose_array.poses.append(pose)

        # Publish the PoseArray
        self.pose_array_publisher.publish(pose_array)
        self.get_logger().info(f"Published PoseArray with {len(pose_array.poses)} poses.")


def main(args=None):
    rclpy.init(args=args)
    
    node = SkeletonPosePublisher()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Node interrupted by user.")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
