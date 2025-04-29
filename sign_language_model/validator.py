import pickle
import cv2
import mediapipe as mp
import numpy as np

# Load the trained model
with open('svm_model.1.0.0.p', 'rb') as f:
    model_dict = pickle.load(f)
    model = model_dict['model']

# MediaPipe initialization
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
hands = mp_hands.Hands(static_image_mode=False, min_detection_confidence=0.3)

# Class Labels (Update this with your actual class names)
CLASS_NAMES = {0: "1", 1: "2", 2: "5", 3: "C", 4: "3"}

# Start capturing video
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame to RGB
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )

            # Extract landmark data
            data_aux, x_, y_ = [], [], []

            for landmark in hand_landmarks.landmark:
                x_.append(landmark.x)
                y_.append(landmark.y)

            for landmark in hand_landmarks.landmark:
                data_aux.append(landmark.x - min(x_))
                data_aux.append(landmark.y - min(y_))

            # Convert to NumPy array & predict
            data_aux = np.array(data_aux).reshape(1, -1)
            prediction = model.predict(data_aux)[0]
            predicted_label = CLASS_NAMES.get(prediction, "Unknown")

            # Display Prediction on Frame
            cv2.putText(frame, predicted_label, (50, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Show frame
    cv2.imshow("Live Gesture Recognition", frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
