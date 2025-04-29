import cv2
import mediapipe as mp
import pickle

NUMBER_OF_CLASSES = 5

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
hands = mp_hands.Hands(static_image_mode=False, min_detection_confidence=0.3)

data, labels = [], []
cap = cv2.VideoCapture(0)

for j in range(NUMBER_OF_CLASSES):
    
    collecting = False
    counter = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if collecting:
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(img_rgb)
            
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        frame,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style(),
                    )
                    data_aux, x_, y_ = [], [], []
                    
                    for landmark in hand_landmarks.landmark:
                        x_.append(landmark.x)
                        y_.append(landmark.y)
                    
                    for landmark in hand_landmarks.landmark:
                        data_aux.append(landmark.x - min(x_))
                        data_aux.append(landmark.y - min(y_))
                    
                    data.append(data_aux)
                    labels.append(j)
                    counter+=1
            cv2.putText(frame, f"Class: {j}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.putText(frame, f"Size: {counter}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        else:
            cv2.putText(frame, "Press Q to start collecting data", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    
        cv2.imshow("Frame", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            if collecting:
                print(f'Stopped collecting data for class {j}')
                break
            else:
                print(f'Started collecting data for class {j}')
                collecting = True

cap.release()
cv2.destroyAllWindows()

with open('data.pickle', 'wb') as f:
    pickle.dump({'data': data, 'labels': labels}, f)
print("Dataset created and saved!")