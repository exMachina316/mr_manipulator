import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseArray, Pose
from std_srvs.srv import Trigger
import pickle
import cv2
import mediapipe as mp
import numpy as np

class HandDrawingNode(Node):
    def __init__(self):
        super().__init__('hand_drawing_node')

        # Declare and get the ROS 2 parameter for the model path
        self.declare_parameter('model_path', "/root/ur_ws/src/mr_manipulator/models/model.2.0.2.p")
        model_path = self.get_parameter('model_path').get_parameter_value().string_value

        # Load the hand gesture model
        self.model_dict = pickle.load(open(model_path, 'rb'))
        self.model = self.model_dict['model']

        sensor_qos = rclpy.qos.qos_profile_sensor_data

        # Create ROS 2 publisher for waypoints as PoseArray
        self.waypoints_publisher = self.create_publisher(PoseArray, 'waypoints', sensor_qos)

        # Create ROS 2 client for executing waypoints
        self.execute_client = self.create_client(Trigger, 'execute_waypoints')

        # Initialize camera
        self.cap = cv2.VideoCapture(0)

        # Initialize Mediapipe Hands
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.hands = self.mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.9)
        
        # Drawing and erasing configurations
        self.drawing_color = (0, 0, 255)
        self.waypoint_color = (0, 255, 0)
        self.canvas = None
        self.waypoints = []
        self.labels_dict = {0: 'Pointer', 1: 'Hold', 2: 'Erase'}

        # Timer for processing frames
        self.create_timer(0.01, self.process_frame)

    def execute_callback(self, future):
        try:
            response = future.result()
            if response.success:
                self.get_logger().info(response.message)
            else:
                self.get_logger().error(response.message)
        except Exception as e:
            self.get_logger().error(f'Service call failed: {str(e)}')

    def process_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            self.get_logger().error("Failed to read from camera.")
            return

        frame = cv2.flip(frame, 1)
        H, W, _ = frame.shape

        if self.canvas is None:
            self.canvas = np.zeros_like(frame)

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(frame_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                self.mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_drawing_styles.get_default_hand_landmarks_style(),
                    self.mp_drawing_styles.get_default_hand_connections_style(),
                )

                data_aux, x_, y_ = [], [], []

                for landmark in hand_landmarks.landmark:
                    x_.append(landmark.x)
                    y_.append(landmark.y)

                for landmark in hand_landmarks.landmark:
                    data_aux.append(landmark.x - min(x_))
                    data_aux.append(landmark.y - min(y_))

                index_x = hand_landmarks.landmark[8].x * W
                index_y = hand_landmarks.landmark[8].y * H

                probabilities = self.model.predict_proba([np.asarray(data_aux)])[0]
                predicted_class = np.argmax(probabilities)
                confidence = probabilities[predicted_class]

                predicted_character = self.labels_dict[predicted_class]

                text = f"{predicted_character} ({confidence:.1%})"
                cv2.putText(frame, text, (int(index_x), int(index_y) - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

                if predicted_character == 'Pointer':
                    if len(self.waypoints) > 1:
                        cv2.line(self.canvas, (int(self.waypoints[-2][0]), int(self.waypoints[-2][1])),
                                 (int(index_x), int(index_y)), self.drawing_color, thickness=5)
                        cv2.circle(self.canvas, (int(index_x), int(index_y)), 3, self.waypoint_color, 3)
                    self.waypoints.append((index_x, index_y))

                elif predicted_character == 'Erase':
                    if self.waypoints:
                        self.get_logger().info("Erasing waypoints.")
                        self.waypoints.clear()
                    self.canvas = np.zeros_like(frame)

                elif predicted_character == 'Hold':
                    if not self.waypoints:
                        self.get_logger().warn("No waypoints to execute")
                        return
                    
                    request = Trigger.Request()
                    future = self.execute_client.call_async(request)
                    future.add_done_callback(self.execute_callback)

        pose_array_msg = PoseArray()
        pose_array_msg.header.stamp = self.get_clock().now().to_msg()
        pose_array_msg.header.frame_id = 'camera_optical_frame'

        if self.waypoints:
            for waypoint in self.waypoints:
                pose = Pose()
                pose.position.x = (waypoint[0] - W/2) * 0.005
                pose.position.y = (waypoint[1] - H/2) * 0.005
                pose.position.z = 1.5
                pose.orientation.w = 1.0
                pose_array_msg.poses.append(pose)

        self.get_logger().debug(f"Published {len(pose_array_msg.poses)} waypoints.")
        self.waypoints_publisher.publish(pose_array_msg)

        frame_with_drawing = cv2.addWeighted(frame, 0.5, self.canvas, 0.5, 0)
        cv2.imshow('Hand Drawing', frame_with_drawing)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            self.cleanup()

    def cleanup(self):
        self.cap.release()
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

if __name__ == '__main__':
    main()
