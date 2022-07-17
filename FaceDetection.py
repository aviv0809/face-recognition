import cv2
import mediapipe as mp
import pickle
def main():
    face_cascade = cv2.CascadeClassifier('cascade/data/haarcascade_frontalface_alt2.xml')

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read("trainner.yml")

    labels = {"person_name": 1}
    with open("label.pickle", 'rb') as f:
        og_labels = pickle.load(f)
        labels = {v: k for k, v in og_labels.items()}

    cap = cv2.VideoCapture(0)
    name = ""
    mpFaceDetection = mp.solutions.face_detection
    mpDraw = mp.solutions.drawing_utils
    faceDetection = mpFaceDetection.FaceDetection(0.75)
    while (True):
        success, img = cap.read()
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        result = faceDetection.process(imgRGB)
        #print(result)
        if result.detections:
            for id, detection in enumerate(result.detections):

                bboxC = detection.location_data.relative_bounding_box
                iH, iW, iC = img.shape
                bbox = int(bboxC.xmin * iW), int(bboxC.ymin * iH), \
                       int(bboxC.width * iW), int(bboxC.height * iH)
                cv2.rectangle(img, bbox, (0, 255, 0), 2)
                #print(id, detection)
                #print(detection.location_data.relative_bounding_box)

        cv2.imshow('frame', img)
        if cv2.waitKey(20) & 0xff == ord('q'):
            break


    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

