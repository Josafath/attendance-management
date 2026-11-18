"""Face detection (YuNet) and face embeddings (SFace), from OpenCV.

How a frame becomes an embedding:

    1. YuNet finds each face: a box, a confidence score and 5 landmarks
       (both eyes, nose tip, both mouth corners).
    2. alignCrop uses the landmarks to rotate and scale the face to a
       standard 112x112 picture, with the eyes always in the same place.
    3. SFace turns that picture into 128 numbers.
    4. The vector is normalized to length 1, so comparing two faces is just a
       dot product (the cosine similarity): 1 = same direction, 0 = unrelated.

`FaceEngine` is a protocol so the tests can replace the neural networks with a
fake engine and test the rest of the logic in milliseconds.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import cv2
import numpy as np

from .models import SFACE, YUNET

# YuNet reports a confidence from 0 to 1 for each face. Below this, it's
# probably not a face (a pattern on a shirt, a poster...).
MIN_DETECTION_SCORE = 0.8


@dataclass(frozen=True)
class Face:
    """A face found in a frame."""
    box: tuple[int, int, int, int]  # x, y, width, height in pixels
    score: float                    # YuNet's confidence that it's a face
    raw: np.ndarray                 # YuNet's full row (box + landmarks), needed by alignCrop

    @property
    def size(self) -> int:
        """The shorter side of the box: small faces are far away and less reliable."""
        return min(self.box[2], self.box[3])


class FaceEngine(Protocol):
    def detect(self, frame: np.ndarray) -> list[Face]: ...
    def embed(self, frame: np.ndarray, face: Face) -> np.ndarray: ...


class OpenCVFaceEngine:
    """The real engine: YuNet + SFace. The models must be downloaded first (models.ensure_models)."""

    def __init__(self, min_score: float = MIN_DETECTION_SCORE) -> None:
        # The input size is a placeholder; detect() sets it to each frame's real size.
        self.detector = cv2.FaceDetectorYN.create(str(YUNET.path()), "", (320, 320),
                                                  score_threshold=min_score)
        self.recognizer = cv2.FaceRecognizerSF.create(str(SFACE.path()), "")

    def detect(self, frame: np.ndarray) -> list[Face]:
        height, width = frame.shape[:2]
        self.detector.setInputSize((width, height))
        _, rows = self.detector.detect(frame)
        if rows is None:  # OpenCV returns None, not an empty array, when there are no faces
            return []
        faces = []
        for row in rows:
            # Row layout: x, y, w, h, 10 landmark coordinates, score (index 14).
            x, y, w, h = (int(round(v)) for v in row[:4])
            faces.append(Face((x, y, w, h), float(row[14]), row))
        return faces

    def embed(self, frame: np.ndarray, face: Face) -> np.ndarray:
        aligned = self.recognizer.alignCrop(frame, face.raw)
        vector = self.recognizer.feature(aligned).reshape(-1)  # (1, 128) -> (128,)
        return normalize(vector)


def normalize(vector: np.ndarray) -> np.ndarray:
    """The same vector with length 1 (float32), so a dot product is the cosine similarity."""
    vector = np.asarray(vector, dtype=np.float32).reshape(-1)
    norm = np.linalg.norm(vector)
    if norm == 0:
        raise ValueError("Can't normalize an all-zero embedding.")
    return vector / norm
