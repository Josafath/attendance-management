import os
import cv2
import numpy as np
from PIL import Image
from tkinter import messagebox


class Trainer:

    def __init__(self):
        self.check_existence_directory()
        self.path = os.path.join(os.getcwd(), "dataset")
        print(self.path)

        self.recognizer = cv2.face.LBPHFaceRecognizer_create()
        self.detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml");

    def train(self):
        print("\n [INFO] Training faces. It will take a few seconds. Wait ...")
        messagebox.showinfo("[INFO]", "Training faces. It will take a few seconds.\n"
                                     "Close window and wait...")

        faces, ids = self.getImagesandIds()
        self.recognizer.train(faces,np.array(ids))
        self.recognizer.write('trainer/trainer.yml')

        print("[INFO] Traning finished with success!")

    def getImagesandIds(self):
        imagePaths = [os.path.join(self.path,f) for f in os.listdir(self.path)]
        facesSample = []
        ids = []
        for imagePath in imagePaths:

            PIL_img = Image.open(imagePath)
            img_numpy = np.array(PIL_img,'uint8')

            id = int(os.path.split(imagePath)[-1].split(".")[1])
            faces = self.detector.detectMultiScale(img_numpy)

            for (x,y,w,h) in faces:
                facesSample.append(img_numpy[y:y+h,x:x+w])
                ids.append(id)


        return facesSample, ids

    def check_existence_directory(self):
        if not os.path.isdir("/trainer"):
            current_workspace = os.getcwd()
            path = os.path.join(current_workspace, "trainer")
            os.mkdir(path)