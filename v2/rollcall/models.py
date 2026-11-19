"""The two neural network models, downloaded once from OpenCV's model zoo.

    YuNet  finds faces in an image and five points on each one (eyes, nose, mouth corners)
    SFace  turns an aligned face into an embedding: 128 numbers that describe it

They are not included in the repository because SFace is 37 MB. The first run
downloads them into the data folder and checks their SHA-256 fingerprint, so a
broken or tampered download is never used.
"""
from __future__ import annotations

import hashlib
import os
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

from . import paths

ZOO = "https://github.com/opencv/opencv_zoo/raw/main/models"


@dataclass(frozen=True)
class Model:
    filename: str
    url: str
    sha256: str
    size: int  # bytes, used to show the download progress

    def path(self) -> Path:
        return paths.models_dir() / self.filename

    def is_ready(self) -> bool:
        """True if the file exists and is the right size.

        Checking the size is instant; the SHA-256 is checked once, right after downloading.
        """
        file = self.path()
        return file.is_file() and file.stat().st_size == self.size


YUNET = Model(
    filename="face_detection_yunet_2023mar.onnx",
    url=f"{ZOO}/face_detection_yunet/face_detection_yunet_2023mar.onnx",
    sha256="8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4",
    size=232_589,
)
SFACE = Model(
    filename="face_recognition_sface_2021dec.onnx",
    url=f"{ZOO}/face_recognition_sface/face_recognition_sface_2021dec.onnx",
    sha256="0ba9fbfa01b5270c96627c4ef784da859931e02f04419c829e83484087c34e79",
    size=38_696_353,
)
MODELS = (YUNET, SFACE)

# progress(model, bytes_downloaded, total_bytes): lets the UI show a progress bar.
Progress = Callable[[Model, int, int], None]


class DownloadError(RuntimeError):
    """A model couldn't be downloaded or its fingerprint didn't match."""


def models_ready() -> bool:
    return all(model.is_ready() for model in MODELS)


def ensure_models(progress: Optional[Progress] = None) -> None:
    """Downloads the models that are missing. Does nothing if they're already there."""
    for model in MODELS:
        if not model.is_ready():
            download(model, progress)


def download(model: Model, progress: Optional[Progress] = None) -> Path:
    """Downloads one model, checks its SHA-256 and only then puts it in place."""
    target = model.path()
    # Download to a temporary name first: if the app closes halfway, the
    # half-written file is never mistaken for the real model.
    partial = target.with_suffix(".part")
    sha = hashlib.sha256()
    done = 0
    try:
        with urllib.request.urlopen(model.url, timeout=30) as response, open(partial, "wb") as out:
            while True:
                chunk = response.read(256 * 1024)  # 256 KB at a time
                if not chunk:
                    break
                out.write(chunk)
                sha.update(chunk)
                done += len(chunk)
                if progress:
                    progress(model, done, model.size)
    except OSError as error:  # no internet, timeout, HTTP error, disk full...
        partial.unlink(missing_ok=True)
        raise DownloadError(f"Couldn't download {model.filename}: {error}") from error

    if sha.hexdigest() != model.sha256:
        partial.unlink(missing_ok=True)
        raise DownloadError(f"{model.filename} was downloaded but its fingerprint doesn't match. "
                            "Try again; if it keeps failing, the file may have changed upstream.")
    # os.replace is atomic: the model appears complete or not at all.
    os.replace(partial, target)
    return target
