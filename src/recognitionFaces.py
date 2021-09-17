import cv2
import numpy as np

class Recognition:

    def __init__(self):
        self.recognizer = cv2.face.LBPHFaceRecognizer_create()
        self.faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

        self.recognizer.read('trainer/trainer.yml')

        self.font = cv2.FONT_HERSHEY_COMPLEX

        self.cam = cv2.VideoCapture(0)
        self.cam.set(3, 640)
        self.cam.set(4, 480)

        self.ids_students = []


    def go(self):
        minW = 0.1 * self.cam.get(3)
        minH = 0.1 * self.cam.get(4)

        while True:

            ret,img = self.cam.read()

            gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

            faces = self.faceCascade.detectMultiScale(
                gray,
                scaleFactor= 1.2,
                minNeighbors= 5,
                minSize= (int(minW), (int(minH))),
            )

            for (x,y,w,h) in faces:
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

                id, confidence = self.recognizer.predict(gray[y:y + h, x:x + w])

                if (confidence < 100):
                    self.ids_students.append(id)
                    confidence = "  {0}%".format(round(100 - confidence))
                else:
                    confidence = "  {0}%".format(round(100 - confidence))


                cv2.putText(img, "Next", (x + 5, y - 5), self.font, 1, (255, 255, 255), 2)

            cv2.imshow('Taking Attendance', img)

            k = cv2.waitKey(10) & 0xff  # Press 'ESC' for exiting video
            if k == 27:
                self.finish()
                break

        self.finish()
        return np.unique(self.ids_students)

    def finish(self):
        self.cam.release()
        cv2.destroyAllWindows()