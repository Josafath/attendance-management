"""Reads the camera and analyzes frames in a background thread.

Face recognition takes 20-40 ms per frame. Doing it in Tkinter's thread would
freeze the window, so a FrameWorker does it in the background and the window
asks for the newest result a few times per second:

    worker thread:  open camera -> read frame -> analyze(frame) -> keep it as "latest"
    window thread:  every 30 ms, latest() -> draw it -> save to the database

Only the newest result is kept: if the window is slower than the camera,
old frames are dropped instead of piling up (the video never lags behind).
"""
from __future__ import annotations

import threading
from typing import Callable, Generic, Optional, TypeVar

import numpy as np

from .camera import Camera

T = TypeVar("T")

# How many empty reads in a row mean the camera stopped working (~1 s at 30 fps).
MAX_EMPTY_READS = 30


class FrameWorker(Generic[T]):
    def __init__(self, open_camera: Callable[[], Camera], analyze: Callable[[np.ndarray], T]) -> None:
        self._open_camera = open_camera  # opened inside the thread: it can take a second
        self._analyze = analyze
        self._stop = threading.Event()
        self._lock = threading.Lock()    # protects _latest, shared by both threads
        self._latest: Optional[tuple[np.ndarray, T]] = None
        self._thread = threading.Thread(target=self._run, name="camera", daemon=True)
        self.error: Optional[BaseException] = None  # set if the camera or analyze() failed

    def start(self) -> "FrameWorker[T]":
        self._thread.start()
        return self

    def stop(self, timeout: float = 2.0) -> None:
        """Asks the thread to finish and waits for it, so the camera is released."""
        self._stop.set()
        if self._thread.is_alive() and self._thread is not threading.current_thread():
            self._thread.join(timeout)

    @property
    def running(self) -> bool:
        return self._thread.is_alive()

    def latest(self) -> Optional[tuple[np.ndarray, T]]:
        """The newest (frame, result), or None if there's nothing new since the last call."""
        with self._lock:
            latest, self._latest = self._latest, None
        return latest

    def _run(self) -> None:
        camera = None
        try:
            camera = self._open_camera()
            empty = 0
            while not self._stop.is_set():
                frame = camera.read()
                if frame is None:
                    empty += 1
                    if empty >= MAX_EMPTY_READS:
                        raise RuntimeError("The camera stopped sending images.")
                    continue
                empty = 0
                result = self._analyze(frame)
                with self._lock:
                    self._latest = (frame, result)
        except BaseException as error:  # reported to the window instead of dying silently
            self.error = error
        finally:
            if camera is not None:
                camera.release()
