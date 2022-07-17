import cv2
import numpy as np
import pickle

def main():

    face_cascade = cv2.CascadeClassifier('cascade/data/haarcascade_frontalface_alt2.xml')

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read("trainner.yml")

    labels = {"person_name": 1}
    with open("label.pickle", 'rb') as f:
        og_labels = pickle.load(f)
        labels = {v:k for k, v in og_labels.items()}

    cap = cv2.VideoCapture(0)
    name = ""
    while (True):
        #frame by frame commands
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)
        color = (255, 0, 0)
        stroke = 2
        for (x, y, w, h) in faces:
            #print(x,y,w,h)
            width = x + w
            height = y + h
            roi_gray = gray[y:width, x:height]
            roi_color = frame[y:width, x:height]
            cv2.rectangle(frame, (x, y), (width, height), color, stroke)
            try:
                id_, conf = recognizer.predict(gray[y:width, x:height])
                print(labels[id_], " confidence: ", conf)
                if conf<80:
                    name = labels[id_]


            except:
                break

            font = cv2.FONT_HERSHEY_SIMPLEX
            color = (0, 0, 255)
            stroke = 2
            cv2.putText(frame, name, (x, y), font, 1, color, stroke, cv2.LINE_AA)

        #display
        cv2.imshow('frame', frame)
        if cv2.waitKey(20) & 0xff == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()