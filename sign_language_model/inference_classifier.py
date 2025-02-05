import pickle
import cv2
import mediapipe as mp
import numpy as np


model_dict = pickle.load(open('D:\\mr_manipulator\\models\\model.1.0.0.p', 'rb'))
model = model_dict['model']

cap = cv2.VideoCapture(0)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)

drawing_color = (0, 0, 255)
waypoint_color = (0, 255, 0)
canvas = None 
waypoints = []

eraser_color = (0, 0, 0)
erasing = False 
pointer = 'Pointer'
erase = 'Erase'
# hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

labels_dict = {0: 'Pointer', 1: 'Hold', 2: 'Erase'}

while True:

    data_aux = []
    x_ = []
    y_ = []

    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    H, W, _ = frame.shape

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(frame_rgb)

    if canvas is None:
        canvas = np.zeros_like(frame)
    
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,  # image to draw
                hand_landmarks,  # model output
                mp_hands.HAND_CONNECTIONS,  # hand connections
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())

        for hand_landmarks in results.multi_hand_landmarks:
            index_x = hand_landmarks.landmark[8].x * canvas.shape[1]
            index_y = hand_landmarks.landmark[8].y * canvas.shape[0]

            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y

                x_.append(x)
                y_.append(y)

            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                data_aux.append(x - min(x_))
                data_aux.append(y - min(y_))

        x1 = int(min(x_) * W) - 10
        y1 = int(min(y_) * H) - 10

        x2 = int(max(x_) * W) - 10
        y2 = int(max(y_) * H) - 10

        prediction = model.predict([np.asarray(data_aux)])
        predicted_character = labels_dict[int(prediction[0])]

        #cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
        cv2.putText(frame, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3,
                    cv2.LINE_AA)
        
        if predicted_character == pointer:
            if len(waypoints)>1:
                cv2.line(canvas, (int(waypoints[-2][0]), int(waypoints[-2][1])), (int(index_x), int(index_y)), drawing_color, thickness=5)
                cv2.circle(canvas, (int(index_x), int(index_y)), 3, waypoint_color, 3)
            waypoints.append((index_x, index_y))
        elif predicted_character == erase:
            waypoints.clear()
            canvas = np.zeros_like(frame) 

    frame_with_drawing = cv2.addWeighted(frame, 0.5, canvas, 0.5, 0)    
    cv2.imshow('frame', frame_with_drawing)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

def send_waypoints(waypoints):
    print("Waypoints Sent!!")
