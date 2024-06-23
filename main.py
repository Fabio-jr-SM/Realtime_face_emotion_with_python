import cv2
import matplotlib.pyplot as plt
from deepface import DeepFace

#
faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    cap = cv2.VideoCapture(1)
if not cap.isOpened():
    raise IOError("Cannot open webcam")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Analyze the frame
    results = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
    
    # Extract the first result
    result = results[0]
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(gray, 1.1, 4)
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    font = cv2.FONT_HERSHEY_SIMPLEX

    cv2.putText(frame,
                result['dominant_emotion'],
                (50, 50),
                font, 3,
                (0, 0, 255),
                2,
                cv2.LINE_4)

    cv2.imshow('Demo Video', frame)

    if cv2.waitKey(5) == 27:  # Press 'ESC' to exit
        break

cap.release()
cv2.destroyAllWindows()
