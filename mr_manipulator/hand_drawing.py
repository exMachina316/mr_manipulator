import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseArray, Pose
import pickle
import cv2
import mediapipe as mp
import numpy as np

class HandDrawingNode(Node):
    def __init__(self):
        super().__init__('hand_drawing_node')

        # Declare and get the ROS 2 parameter for the model path
        self.declare_parameter('model_path', "/root/ur_ws/src/mr_manipulator/models/model.p")
        model_path = self.get_parameter('model_path').get_parameter_value().string_value

        # Load the hand gesture model
        self.model_dict = pickle.load(open(model_path, 'rb'))
        self.model = self.model_dict['model']

        # Create ROS 2 publisher for waypoints as PoseArray
        self.waypoints_publisher = self.create_publisher(PoseArray, 'waypoints', 10)

        # Initialize camera
        self.cap = cv2.VideoCapture(0)

        # Initialize Mediapipe Hands
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.hands = self.mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)

        # Drawing and erasing configurations
        self.drawing_color = (0, 0, 255)
        self.waypoint_color = (0, 255, 0)
        self.canvas = None
        self.waypoints = []
        self.labels_dict = {0: 'Pointer', 1: 'Hold', 2: 'Erase'}

        # Timer for processing frames
        self.create_timer(0.01, self.process_frame)

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

                prediction = self.model.predict([np.asarray(data_aux)])
                predicted_character = self.labels_dict[int(prediction[0])]

                cv2.putText(frame, predicted_character, (int(index_x), int(index_y) - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

                if predicted_character == 'Pointer':
                    if len(self.waypoints) > 1:
                        cv2.line(self.canvas, (int(self.waypoints[-2][0]), int(self.waypoints[-2][1])),
                                 (int(index_x), int(index_y)), self.drawing_color, thickness=5)
                        cv2.circle(self.canvas, (int(index_x), int(index_y)), 3, self.waypoint_color, 3)
                    self.waypoints.append((index_x, index_y))

                elif predicted_character == 'Erase':
                    self.waypoints.clear()
                    self.canvas = np.zeros_like(frame)

        frame_with_drawing = cv2.addWeighted(frame, 0.5, self.canvas, 0.5, 0)
        cv2.imshow('Hand Drawing', frame_with_drawing)

        # Publish waypoints to the ROS 2 topic as PoseArray
        if self.waypoints:
            pose_array_msg = PoseArray()
            pose_array_msg.header.stamp = self.get_clock().now().to_msg()
            pose_array_msg.header.frame_id = 'camera_frame'

            for waypoint in self.waypoints:
                pose = Pose()
                pose.position.x = waypoint[0]
                pose.position.y = waypoint[1]
                pose.position.z = 0.0
                pose.orientation.w = 1.0
                pose_array_msg.poses.append(pose)

            self.waypoints_publisher.publish(pose_array_msg)

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
