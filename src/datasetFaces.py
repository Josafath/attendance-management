import cv2
from tkinter import messagebox
import os

class DatasetFaces:

    def __init__(self):
        self.check_existence_directory()
        self.cam =cv2.VideoCapture(0)
        self.cam.set(3,640)
        self.cam.set(4,480)
        self.face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    def start(self,id_student):
        print("\n [INFO] Initializing face capture. Look the camera and wait ...")
        messagebox.showinfo("Starting","[INFO] Initializing face capture.\nClose this message, look the camera and wait...")
        dummy_counter = 0
        while True:
            ret, img = self.cam.read()
            gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            faces = self.face_detector.detectMultiScale(gray,1.3,5)

            for (x,y,w,h) in faces:

                cv2.rectangle(img,(x,y),(x+w,y+h), (255,0,0), 2)
                dummy_counter += 1
                cv2.imwrite("dataset/Student." + str(id_student) + '.' + str(dummy_counter) + ".jpg", gray[y:y+h,x:x+w])


                cv2.imshow("image",img)

            k = cv2.waitKey(100) & 0xff  # Esc key to close the process
            if k == 27:
                break
            # The program will only take 20 pictures per user
            elif dummy_counter >= 20:
                break

        self.finish()

    def finish(self):
        messagebox.showinfo("","Student Added Successfully.")
        self.cam.release()
        cv2.destroyAllWindows()


    def check_existence_directory(self):
        if not os.path.isdir("dataset"):
            current_workspace = os.getcwd()
            path = os.path.join(current_workspace, "dataset")
            os.mkdir(path)
