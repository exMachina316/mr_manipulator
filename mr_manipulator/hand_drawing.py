import os
import cv2
import numpy as np

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseArray, Pose
from std_srvs.srv import Trigger
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
from mr_manipulator import mp_gesture_live


class HandDrawingNode(Node):
    def __init__(self):
        super().__init__("hand_drawing_node")

        # Setup Parameters
        self.declare_parameter("model_path", "")
        self.declare_parameter("camera_topic", "/oak/rgb/image_rect")
        self.declare_parameter("camera_info_topic", "/oak/rgb/camera_info")

        model_path = self.get_parameter("model_path").get_parameter_value().string_value
        camera_topic = (
            self.get_parameter("camera_topic").get_parameter_value().string_value
        )
        camera_info_topic = (
            self.get_parameter("camera_info_topic").get_parameter_value().string_value
        )

        if not os.path.exists(model_path):
            self.get_logger().error(f"Model file not found at: {model_path}")
            raise FileNotFoundError(f"Model file not found at: {model_path}")

        # Create resources
        self.camera_info = None
        self.live_gesture_recognizer = mp_gesture_live.LiveGestureRecognizer(model_path=model_path)
        self.bridge = CvBridge()

        sensor_qos = rclpy.qos.qos_profile_sensor_data
        self.waypoints_publisher = self.create_publisher(
            PoseArray, "waypoints", sensor_qos
        )
        self.execute_client = self.create_client(Trigger, "execute_waypoints")

        # Create a subscriber to the image topic
        self.image_subscription = self.create_subscription(
            Image, camera_topic, self.image_callback, 10
        )

        self.camera_info_subscription = self.create_subscription(
            CameraInfo, camera_info_topic, self.camera_info_callback, 10
        )

        # Drawing and erasing configurations
        self.drawing_color = (0, 0, 255)
        self.waypoint_color = (0, 255, 0)
        self.canvas = None

        self.waypoints = []
        self.gesture_label_map = {"Pointing_Up": "Pointer", "Open_Palm": "Hold", "Victory": "Erase", "Thumbs_Up": "Execute"}

        # Hold gesture tracking
        self.execute_start_time = None
        self.execute_triggered = False

    def camera_info_callback(self, msg):
        self.camera_info = msg
        self.get_logger().info("Camera info received.")
        # We can unsubscribe after receiving it once if it's static
        self.destroy_subscription(self.camera_info_subscription)
        self.camera_info_subscription = None

    def execute_callback(self, future):
        try:
            response = future.result()
            if response.success:
                self.get_logger().info(response.message)
            else:
                self.get_logger().error(response.message)
        except Exception as e:
            self.get_logger().error(f"Service call failed: {str(e)}")

    def image_callback(self, msg):
        if self.camera_info is None:
            self.get_logger().warn("Waiting for camera info...")
            return

        try:
            frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception as e:
            self.get_logger().error(f"Failed to convert image: {e}")
            return

        frame = cv2.flip(frame, 1)
        H, W, _ = frame.shape

        if self.canvas is None:
            self.canvas = np.zeros_like(frame)

            landmarked_frame, score, gesture_code, hand_landmarks, _ = self.live_gesture_recognizer.run_live_inference(frame)

            try:
                gesture_name = self.gesture_label_map[gesture_code]
            except KeyError:
                gesture_name = "Unknown"

            if gesture_name == "Pointer":
                self.execute_start_time = None  # Reset hold timer
                self.execute_triggered = False
                index_x, index_y = hand_landmarks[8].x, hand_landmarks[8].y
                if len(self.waypoints) > 1:
                    cv2.line(
                        self.canvas,
                        (int(self.waypoints[-2][0]), int(self.waypoints[-2][1])),
                        (int(index_x), int(index_y)),
                        self.drawing_color,
                        thickness=5,
                    )
                    cv2.circle(
                        self.canvas,
                        (int(index_x), int(index_y)),
                        3,
                        self.waypoint_color,
                        3,
                    )
                self.waypoints.append((index_x, index_y))

            elif gesture_name == "Erase":
                self.execute_start_time = None  # Reset hold timer
                self.execute_triggered = False
                if self.waypoints:
                    self.get_logger().info("Erasing waypoints.")
                    self.waypoints.clear()
                self.canvas = np.zeros_like(frame)

            elif gesture_name == "Execute":

                if not self.waypoints:
                    self.get_logger().warn("No waypoints to execute")
                    return

                if self.execute_start_time is None:
                    self.execute_start_time = self.get_clock().now()

                execute_duration= (
                    self.get_clock().now() - self.execute_start_time
                ).nanoseconds / 1e9

                if execute_duration >= 2.0 and not self.execute_triggered:
                    self.get_logger().info(
                        "Hold detected for 2 seconds, executing waypoints."
                    )
                    request = Trigger.Request()
                    future = self.execute_client.call_async(request)
                    future.add_done_callback(self.execute_callback)
                    self.execute_triggered = True  # Prevent re-triggering

        if self.execute_triggered:
            cv2.putText(frame, "Executing...", (W//2, H//2), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)

        pose_array_msg = PoseArray()
        pose_array_msg.header.stamp = self.get_clock().now().to_msg()
        pose_array_msg.header.frame_id = msg.header.frame_id

        if self.waypoints:
            fx = self.camera_info.k[0]
            fy = self.camera_info.k[4]
            cx = self.camera_info.k[2]
            cy = self.camera_info.k[5]

            for waypoint in self.waypoints:
                u = waypoint[0]
                v = waypoint[1]

                pose = Pose()

                # Convert pixel coordinates to camera frame coordinates
                z = 1.015  # a fixed depth
                x = (u - cx) * z / fx
                y = (v - cy) * z / fy

                pose.position.x = x
                pose.position.y = y
                pose.position.z = z
                pose.orientation.w = 1.0
                pose_array_msg.poses.append(pose)

        self.get_logger().debug(f"Published {len(pose_array_msg.poses)} waypoints.")
        self.waypoints_publisher.publish(pose_array_msg)

        frame_with_drawing = cv2.addWeighted(landmarked_frame, 0.5, self.canvas, 0.5, 0)
        cv2.imshow("Hand Drawing", frame_with_drawing)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            self.cleanup()

    def cleanup(self):
        cv2.destroyAllWindows()
        self.get_logger().info("Shutting down Hand Drawing Node.")
        self.destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = HandDrawingNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.cleanup()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
