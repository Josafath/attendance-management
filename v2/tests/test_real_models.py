"""End-to-end check with the REAL YuNet and SFace models and real photos.

Skipped unless ROLLCALL_TEST_ASSETS points to a folder with:
    face_detection_yunet_2023mar.onnx, face_recognition_sface_2021dec.onnx
    messi5.jpg, lena.jpg   (sample images from github.com/opencv/opencv/tree/4.x/samples/data)

Photos of people are deliberately not stored in this repository.
"""
import os
import shutil
from pathlib import Path

import cv2
import numpy as np
import pytest

from rollcall import models
from rollcall.faces import OpenCVFaceEngine
from rollcall.recognition import AttendanceTracker, EnrollStatus, Enroller, Matcher

ASSETS = os.environ.get("ROLLCALL_TEST_ASSETS")
pytestmark = pytest.mark.skipif(not ASSETS, reason="set ROLLCALL_TEST_ASSETS to run the real-model tests")


@pytest.fixture
def engine():
    for model in models.MODELS:
        shutil.copy(Path(ASSETS) / model.filename, model.path())  # into the test's data folder
    assert models.models_ready()
    return OpenCVFaceEngine()


def image(name: str) -> np.ndarray:
    return cv2.imread(str(Path(ASSETS) / name))


def variants(img: np.ndarray) -> list[np.ndarray]:
    """The same photo as a camera might see it: mirrored, darker, brighter, slightly rotated."""
    h, w = img.shape[:2]
    out = []
    for angle in (-8, -4, 0, 4, 8):
        rotation = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        rotated = cv2.warpAffine(img, rotation, (w, h))
        out += [rotated, cv2.flip(rotated, 1), cv2.convertScaleAbs(rotated, alpha=0.6, beta=10),
                cv2.convertScaleAbs(rotated, alpha=1.3, beta=-10)]
    return out


def test_detects_one_face_in_each_photo(engine):
    for name in ("messi5.jpg", "lena.jpg"):
        faces = engine.detect(image(name))
        assert len(faces) == 1, name
        assert np.linalg.norm(engine.embed(image(name), faces[0])) == pytest.approx(1.0, abs=1e-5)


def test_register_then_recognize(engine):
    messi, lena = image("messi5.jpg"), image("lena.jpg")
    # The photos' faces are small (~30 px in messi5.jpg), so the size limits are lowered.
    enroller = Enroller(engine, needed=5, every=1, min_face_size=20)
    for frame in variants(messi):
        if enroller.process(frame).status == EnrollStatus.DONE:
            break
    assert enroller.done

    matcher = Matcher(np.full(len(enroller.embeddings), 1), np.array(enroller.embeddings))
    tracker = AttendanceTracker(engine, matcher, needed=3, window=5, min_face_size=20)

    confirmed = []
    for frame in variants(messi)[:6]:
        confirmed += tracker.process(frame).newly_confirmed
    assert [sid for sid, _ in confirmed] == [1]

    # A different person is never taken for the registered one.
    for frame in variants(lena):
        result = tracker.process(frame)
        assert all(face.student_id is None for face in result.faces)
        assert result.newly_confirmed == []
