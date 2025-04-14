import cv2
import mediapipe as mp

# Initialize Mediapipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)

# Capture video from webcam
cap = cv2.VideoCapture(0)

# Tip landmark IDs for all 5 fingers
tip_ids = [4, 8, 12, 16, 20]

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Flip frame horizontally for mirror view
    frame = cv2.flip(frame, 1)

    # Convert to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame with Mediapipe
    results = hands.process(rgb_frame)

    # Draw hand landmarks and count fingers
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            landmarks = hand_landmarks.landmark
            fingers = []

            # Thumb: compare x-coordinate (different logic due to orientation)
            if landmarks[tip_ids[0]].x < landmarks[tip_ids[0] - 1].x:
                fingers.append(1)
            else:
                fingers.append(0)

            # Fingers: compare y-coordinate of tip and joint 2 below
            for id in range(1, 5):
                if landmarks[tip_ids[id]].y < landmarks[tip_ids[id] - 2].y:
                    fingers.append(1)
                else:
                    fingers.append(0)

            total_fingers = fingers.count(1)

            # Show number of fingers on screen
            cv2.putText(frame, f"Fingers: {total_fingers}", (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)

    # Display the output frame
    cv2.imshow("Hand Counter", frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up
cap.release()
cv2.destroyAllWindows()
