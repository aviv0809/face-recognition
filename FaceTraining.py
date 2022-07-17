import os
import cv2
import numpy as np
from PIL import Image
import pickle
def main():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    image_dir = os.path.join(BASE_DIR, "images")

    face_casade = cv2.CascadeClassifier('cascade/data/haarcascade_frontalface_alt2.xml')
    recognizer = cv2.face.LBPHFaceRecognizer_create()


    current_id = 0
    label_ids = {}
    y_labels = []
    x_train = []

    for root, dirs, files in os.walk(image_dir):
        for file in files:
            if file.endswith("png") or file.endswith("jpg"):
                path = os.path.join(root, file)
                label = os.path.basename(os.path.dirname(path)).replace(" ", "-").lower()
                #print(label, path)
                if not label in label_ids:

                    label_ids[label] = current_id
                    current_id += 1
                print(current_id)
                id_ = label_ids[label]
                print(label_ids)


                pil_image = Image.open(path).convert("L") #turn into grayscale
                image_array = np.array(pil_image, "uint8")
                #print(image_array)
                faces = face_casade.detectMultiScale(image_array, scaleFactor=1.1, minNeighbors=3)
                for (x, y, w, h) in faces:
                    roi = image_array[y:y+h, x:x+w]
                    x_train.append(roi)
                    y_labels.append(id_)
    #print(y_labels)
    print(x_train)

    with open("label.pickle", 'wb') as f:
        pickle.dump(label_ids, f)
    #print(np.array(y_labels))
    recognizer.train(x_train, np.array(y_labels))
    recognizer.save("trainner.yml")