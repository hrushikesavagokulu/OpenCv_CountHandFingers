from flask import Flask, render_template, Response
import cv2
import mediapipe as mp

app = Flask(__name__)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
tip_ids = [4, 8, 12, 16, 20]
cap = cv2.VideoCapture(0)

def gen_frames():
    while True:
        success, frame = cap.read()
        if not success:
            break
        else:
            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb_frame)

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                    landmarks = hand_landmarks.landmark
                    fingers = []

                    if landmarks[tip_ids[0]].x < landmarks[tip_ids[0] - 1].x:
                        fingers.append(1)
                    else:
                        fingers.append(0)

                    for id in range(1, 5):
                        if landmarks[tip_ids[id]].y < landmarks[tip_ids[id] - 2].y:
                            fingers.append(1)
                        else:
                            fingers.append(0)

                    total_fingers = fingers.count(1)

                    cv2.putText(frame, f"Fingers: {total_fingers}", (10, 70),
                                cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)

            _, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/video')
def video():
    return Response(gen_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
