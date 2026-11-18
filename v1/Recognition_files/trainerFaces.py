import os
import cv2
import numpy as np
from PIL import Image

from Recognition_files.datasetFaces import FILES_DIR, DATASET_DIR

# Fix: absolute paths, so it works from any working directory
TRAINER_DIR = os.path.join(FILES_DIR, "trainer")
TRAINER_FILE = os.path.join(TRAINER_DIR, "trainer.yml")


class Trainer:

    def __init__(self):
        self.check_existence_directory()
        self.path = DATASET_DIR

        self.recognizer = cv2.face.LBPHFaceRecognizer_create()

    def train(self):
        """Trains the model with every picture in the dataset. Returns False if there are none."""
        print("\n [INFO] Training faces. It will take a few seconds. Wait ...")

        faces, ids = self.getImagesandIds()
        # Fix: training with no pictures crashed the program
        if not faces:
            return False

        self.recognizer.train(faces,np.array(ids))
        self.recognizer.write(TRAINER_FILE)

        print("[INFO] Traning finished with success!")
        return True

    def getImagesandIds(self):
        # Fix: only read the pictures saved by DatasetFaces (Student.<id>.<n>.jpg)
        imagePaths = [os.path.join(self.path,f) for f in os.listdir(self.path)
                      if f.startswith("Student.") and f.endswith(".jpg")]
        facesSample = []
        ids = []
        for imagePath in imagePaths:

            PIL_img = Image.open(imagePath).convert('L')  # grayscale
            img_numpy = np.array(PIL_img,'uint8')

            id = int(os.path.split(imagePath)[-1].split(".")[1])

            # Fix: the pictures are already cropped faces, so they are used directly.
            # Before, the face detector ran again on each crop and often found
            # nothing, so many pictures (or all of them) were lost.
            facesSample.append(img_numpy)
            ids.append(id)

        return facesSample, ids

    def check_existence_directory(self):
        os.makedirs(TRAINER_DIR, exist_ok=True)
        os.makedirs(DATASET_DIR, exist_ok=True)
