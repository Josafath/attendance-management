"""Where the images come from.

The rest of the app only knows the `Camera` protocol: something with read()
and release(). The real app uses `Webcam`; the tests use a fake camera that
returns prepared images, so the recognition can be tested without a webcam.
"""
from __future__ import annotations

from typing import Optional, Protocol

import cv2
import numpy as np


class Camera(Protocol):
    def read(self) -> Optional[np.ndarray]:
        """The next frame as a BGR image (OpenCV's color order), or None if there isn't one."""

    def release(self) -> None:
        """Frees the device so other programs can use it."""


class CameraError(RuntimeError):
    """The webcam couldn't be opened."""


class Webcam:
    """A real webcam, through OpenCV."""

    def __init__(self, index: int = 0, width: int = 640, height: int = 480) -> None:
        self.capture = cv2.VideoCapture(index)
        if not self.capture.isOpened():
            # Fail right away with a clear message instead of returning empty frames.
            raise CameraError(f"Couldn't open camera {index}. Is it connected, "
                              "or is another program using it?")
        # The camera may not support this size; it then picks the closest one.
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    def read(self) -> Optional[np.ndarray]:
        ok, frame = self.capture.read()
        return frame if ok else None

    def release(self) -> None:
        self.capture.release()

    # "with Webcam() as cam:" always releases the camera, even after an error.
    def __enter__(self) -> "Webcam":
        return self

    def __exit__(self, *exc) -> None:
        self.release()
