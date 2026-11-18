import cv2
import numpy as np

from Recognition_files.datasetFaces import CASCADE_FILE
from Recognition_files.trainerFaces import TRAINER_FILE

# LBPH returns a distance, not a percentage: 0 = identical, bigger = more different.
# Fix: before, any face with a distance under 100 was accepted, so people who were
# not registered were marked as present. Lower it if strangers are still accepted,
# raise it if registered students are not recognized.
MAX_DISTANCE = 70

class Recognition:

    def __init__(self):
        self.recognizer = cv2.face.LBPHFaceRecognizer_create()
        self.faceCascade = cv2.CascadeClassifier(CASCADE_FILE)

        self.recognizer.read(TRAINER_FILE)

        self.font = cv2.FONT_HERSHEY_COMPLEX

        self.cam = cv2.VideoCapture(0)
        self.cam.set(3, 640)
        self.cam.set(4, 480)

        self.id_students = []


    def go(self):
        minW = 0.1 * self.cam.get(3)
        minH = 0.1 * self.cam.get(4)

        while True:

            ret,img = self.cam.read()
            # Fix: stop if the camera doesn't give an image (before, it crashed)
            if not ret:
                break

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            faces = self.faceCascade.detectMultiScale(
                gray,
                scaleFactor= 1.2,
                minNeighbors= 5,
                minSize= (int(minW), (int(minH))),
            )

            for (x,y,w,h) in faces:
                id, distance = self.recognizer.predict(gray[y:y + h, x:x + w])

                if distance < MAX_DISTANCE:
                    self.id_students.append(int(id))
                    # Fix: show who was recognized (green) instead of "Check" for everyone
                    color, text = (0, 255, 0), f"Id {id}"
                else:
                    color, text = (0, 0, 255), "Unknown"

                cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
                cv2.putText(img, text, (x + 5, y - 5), self.font, 1, color, 2)

            cv2.imshow('Taking Attendance (ESC to finish)', img)

            k = cv2.waitKey(10) & 0xff  # Press 'ESC' for exiting video
            if k == 27:
                break

        # Fix: finish() was called twice when ESC was pressed
        self.finish()
        return [int(id) for id in np.unique(self.id_students)]

    def finish(self):
        self.cam.release()
        cv2.destroyAllWindows()
