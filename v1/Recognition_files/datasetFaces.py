import cv2
from tkinter import messagebox
import os

# Fix: paths are built from this file's folder, so the program works
# from any working directory (before, it only worked when run from src/).
FILES_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(FILES_DIR, "dataset")
CASCADE_FILE = os.path.join(FILES_DIR, "haarcascade_frontalface_default.xml")

class DatasetFaces:

    def __init__(self):
        self.check_existence_directory()
        self.cam = cv2.VideoCapture(0)
        self.cam.set(3, 640)
        self.cam.set(4, 480)
        self.face_detector = cv2.CascadeClassifier(CASCADE_FILE)

    def start(self,id_student):
        """Takes up to 20 pictures of the student's face and returns how many were saved."""
        print("\n [INFO] Initializing face capture. Look the camera and wait ...")
        messagebox.showinfo("Starting",
                            "[INFO] Initializing face capture.\n"
                            "Close this message, look the camera and wait...")
        dummy_counter = 0
        while True:
            ret, img = self.cam.read()
            # Fix: stop if the camera doesn't give an image (before, it crashed)
            if not ret:
                messagebox.showerror("Camera", "The camera could not be opened.")
                break

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = self.face_detector.detectMultiScale(gray)

            # Fix: only save a picture when there is exactly one face in the image.
            # Before, if two people were in front of the camera, both faces were
            # saved with the same student id.
            if len(faces) == 1:
                (x,y,w,h) = faces[0]
                cv2.rectangle(img,(x,y),(x+w,y+h), (255,0,0), 2)
                dummy_counter += 1
                cv2.imwrite(os.path.join(DATASET_DIR, "Student." +
                            str(id_student) + '.' +
                            str(dummy_counter) + ".jpg"), gray[y:y+h,x:x+w])

            # Fix: show the camera all the time, not only when a face is found
            cv2.imshow("image",img)

            k = cv2.waitKey(100) & 0xff  # Esc key to close the process
            if k == 27:
                break
            # The program will only take 20 pictures per user
            elif dummy_counter >= 20:
                break

        self.finish()
        return dummy_counter

    def finish(self):
        self.cam.release()
        cv2.destroyAllWindows()


    def check_existence_directory(self):
        # Fix: makedirs with exist_ok doesn't fail if the folder already exists
        os.makedirs(DATASET_DIR, exist_ok=True)
