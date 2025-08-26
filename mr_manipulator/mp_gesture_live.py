import cv2
import mediapipe as mp
import time
import os
from mediapipe.framework.formats import landmark_pb2

# MediaPipe imports
BaseOptions = mp.tasks.BaseOptions
GestureRecognizer = mp.tasks.vision.GestureRecognizer
GestureRecognizerOptions = mp.tasks.vision.GestureRecognizerOptions
GestureRecognizerResult = mp.tasks.vision.GestureRecognizerResult
VisionRunningMode = mp.tasks.vision.RunningMode

class LiveGestureRecognizer:
    def __init__(self, model_path=None):
        # Use the gesture_recognizer.task file in the current directory if no path provided
        if model_path is None:
            raise FileNotFoundError("Model path must be provided.")            

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at: {model_path}")

        self.model_path = model_path
        self.latest_result = None
        self.latest_timestamp = 0
        self.fps = 0

        # Initialize MediaPipe drawing utilities
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles

        # Create gesture recognizer
        self.recognizer = self._create_recognizer()

    def _create_recognizer(self):
        """Create and return a gesture recognizer instance"""
        base_options = BaseOptions(model_asset_path=self.model_path)
        options = GestureRecognizerOptions(
            base_options=base_options,
            running_mode=VisionRunningMode.LIVE_STREAM,
            result_callback=self._result_callback
        )
        return GestureRecognizer.create_from_options(options)

    def _result_callback(self, result, output_image, timestamp_ms):
        """Callback function to handle gesture recognition results"""
        self.latest_result = result
        self.fps = (timestamp_ms-self.latest_timestamp)
        self.latest_timestamp = timestamp_ms

    def draw_landmarks_and_gestures(self, image, result):
        """Draw hand landmarks and gesture labels on the image"""
        if not result:
            return image

        # Draw hand landmarks
        if result.gestures:
            for i, gesture_list in enumerate(result.gestures):
                if gesture_list:
                    top_gesture = gesture_list[0]
                    score = top_gesture.score
                    gesture_name = top_gesture.category_name

                    if result.hand_landmarks and i < len(result.hand_landmarks):
                        landmarks = result.hand_landmarks[i]
                        hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
                        hand_landmarks_proto.landmark.extend([
                            landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in landmarks
                        ])

                        self.mp_drawing.draw_landmarks(
                            image,
                            hand_landmarks_proto,
                            self.mp_hands.HAND_CONNECTIONS,
                            self.mp_drawing_styles.get_default_hand_landmarks_style(),
                            self.mp_drawing_styles.get_default_hand_connections_style(),
                        )

                    label = f"FPS: {self.fps}: {gesture_name} ({score:.2f})"
                    cv2.putText(image, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        return image

    def run_live_inference(self, camera_id=0, window_name="Gesture Recognition"):
        """Run live gesture recognition from camera feed"""
        cap = cv2.VideoCapture(camera_id)

        if not cap.isOpened():
            print(f"Error: Could not open camera {camera_id}")
            return

        print("Starting live gesture recognition. Press 'q' to quit.")

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("Error: Could not read frame from camera")
                    break

                # Flip frame horizontally for mirror effect
                frame = cv2.flip(frame, 1)

                # Convert BGR to RGB for MediaPipe
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # Create MediaPipe Image
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

                # Get timestamp in milliseconds
                timestamp_ms = int(time.time() * 1000)

                # Perform async gesture recognition
                self.recognizer.recognize_async(mp_image, timestamp_ms)

                # Draw landmarks and gestures on the frame
                annotated_frame = self.draw_landmarks_and_gestures(frame, self.latest_result)

                # Display the frame
                cv2.imshow(window_name, annotated_frame)

                # Break loop on 'q' key press
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

                # Small delay to prevent overwhelming the processor
                time.sleep(0.01)

        except KeyboardInterrupt:
            print("\nInterrupted by user")

        finally:
            cap.release()
            cv2.destroyAllWindows()
            self.recognizer.close()

    def get_latest_result(self):
        """Get the latest gesture recognition result"""
        return self.latest_result


def main():
    """Main function to run the live gesture recognizer"""
    try:
        # Initialize the gesture recognizer
        gesture_recognizer = LiveGestureRecognizer()

        # Run live inference
        gesture_recognizer.run_live_inference()

    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please ensure the gesture_recognizer.task file is in the same directory as this script.")
    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    main()
