import numpy as np
import cv2
import imutils
import time
from keras.preprocessing.image import img_to_array
from keras.models import load_model
import numpy as np

detection_model_path = 'haarcascade_files/haarcascade_frontalface_default.xml'
emotion_model_path = 'models/_mini_XCEPTION.35-0.62.hdf5'

emotion_classifier = load_model(emotion_model_path, compile=False)
EMOTIONS = ["angry" ,"disgust","scared", "happy", "sad", "surprised", "neutral"]

face_detection = cv2.CascadeClassifier(detection_model_path)
camera = cv2.VideoCapture("C:/Users/TuPM/Downloads/Video/emotion_video.mkv")

cv2.namedWindow('frame')

while True:
    start_time = time.time()
    ret, frame = camera.read()

    frame = imutils.resize(frame,width=300)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_detection.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=5,minSize=(30,30),flags=cv2.CASCADE_SCALE_IMAGE)

    if len(faces) > 0:
        # for (x,y,w,h) in faces:
        #     cv2.rectangle(gray, (x, y), (x+w, y+h), (0, 255, 0), 2)
        faces = sorted(faces, reverse=True, key=lambda x: (x[2] - x[0]) * (x[3] - x[1]))[0]

    cv2.imshow('frame', gray)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    # print("FPS: ", 1.0 / (time.time() - start_time))
    # print(len(faces))
    

camera.release()
cv2.distroyAllWindows()