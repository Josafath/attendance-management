"""Showing camera frames in Tkinter, without Pillow.

Tkinter's PhotoImage can read PPM, one of the simplest image formats:

    b"P6 640 480 255 " + the red, green and blue bytes of every pixel

OpenCV gives frames as NumPy arrays in blue-green-red order, so converting a
frame is just reversing the color channels and adding that header (~1 ms).

Names are drawn on the Canvas, not on the image: Tk fonts handle accents
("José", "Peña"), while OpenCV's putText only draws plain ASCII.
"""
from __future__ import annotations

import gc
import tkinter as tk
from dataclasses import dataclass
from typing import Callable, Optional

import cv2
import numpy as np

from ..worker import FrameWorker
from . import theme


def to_ppm(frame: np.ndarray) -> bytes:
    """A BGR frame as PPM bytes for tk.PhotoImage."""
    height, width = frame.shape[:2]
    rgb = np.ascontiguousarray(frame[:, :, ::-1])  # BGR -> RGB, as one block of memory
    return b"P6 %d %d 255 " % (width, height) + rgb.tobytes()


def fit(frame: np.ndarray, width: int, height: int) -> tuple[np.ndarray, float]:
    """Scales a frame to fit in width x height, keeping its shape. Returns (frame, scale)."""
    h, w = frame.shape[:2]
    scale = min(width / w, height / h)
    if abs(scale - 1) > 0.01:
        # INTER_AREA gives the best quality when shrinking.
        interpolation = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
        frame = cv2.resize(frame, (max(1, round(w * scale)), max(1, round(h * scale))),
                           interpolation=interpolation)
    return frame, scale


@dataclass(frozen=True)
class Box:
    """A rectangle to draw over the video, in the frame's pixel coordinates."""
    x: int
    y: int
    w: int
    h: int
    color: str
    label: str = ""


class VideoCanvas(tk.Canvas):
    """Shows the latest camera frame with labeled boxes on top."""

    def __init__(self, parent: tk.Misc, width: int = 640, height: int = 480) -> None:
        super().__init__(parent, width=width, height=height, bg=theme.VIDEO_BG, highlightthickness=0)
        self.size = (width, height)
        self._photo: Optional[tk.PhotoImage] = None  # keep a reference or Tk shows nothing
        self.message("Camera off")

    def message(self, text: str) -> None:
        """Clears the video and shows a centered message."""
        self.delete("all")
        self._photo = None
        self.create_text(self.size[0] // 2, self.size[1] // 2, text=text, fill="white",
                         font=(theme.FONT, 14))

    def show(self, frame: np.ndarray, boxes: list[Box] = ()) -> None:
        width, height = self.size
        frame, scale = fit(frame, width, height)
        # Center the image: offsets are the empty margins on each side.
        dx, dy = (width - frame.shape[1]) // 2, (height - frame.shape[0]) // 2
        self.delete("all")
        self._photo = tk.PhotoImage(data=to_ppm(frame), format="PPM")
        self.create_image(dx, dy, anchor="nw", image=self._photo)
        for box in boxes:
            x0, y0 = dx + box.x * scale, dy + box.y * scale
            x1, y1 = x0 + box.w * scale, y0 + box.h * scale
            self.create_rectangle(x0, y0, x1, y1, outline=box.color, width=3)
            if box.label:
                text = self.create_text(x0 + 4, y0 - 4, text=box.label, anchor="sw",
                                        fill="white", font=(theme.FONT, 11, "bold"))
                # A colored background behind the text, so it's readable on any image.
                left, top, right, bottom = self.bbox(text)
                background = self.create_rectangle(left - 4, top - 2, right + 4, bottom + 2,
                                                   fill=box.color, outline="")
                self.tag_lower(background, text)


class WorkerPoller:
    """Checks a FrameWorker every `interval` ms from Tkinter's thread.

    Tkinter isn't thread-safe: only its own thread may touch widgets. So the
    worker never calls the window; the window asks the worker with after().

    It also runs Python's garbage collector from Tkinter's thread now and then.
    Closed windows can linger in reference cycles until the collector frees
    them, and the collector runs in whichever thread happens to be allocating
    memory: often the camera thread. Freeing Tk objects from that thread makes
    Tkinter call into the wrong thread. Collecting here first avoids it.
    """

    GC_EVERY = 60  # polls, about every 2 seconds at 30 ms

    def __init__(self, widget: tk.Misc, worker: FrameWorker,
                 on_result: Callable, on_error: Callable[[BaseException], None], interval: int = 30) -> None:
        self.widget, self.worker = widget, worker
        self.on_result, self.on_error = on_result, on_error
        self.interval = interval
        self._job: Optional[str] = None
        self._active = False
        self._polls = 0

    def start(self) -> "WorkerPoller":
        self._active = True
        gc.collect()  # free closed windows here, before the camera thread exists
        self.worker.start()
        self._job = self.widget.after(self.interval, self._poll)
        return self

    def stop(self) -> None:
        """Stops polling and the worker. Safe to call more than once, even from on_result."""
        self._active = False
        if self._job is not None:
            self.widget.after_cancel(self._job)
            self._job = None
        self.worker.stop()

    def _poll(self) -> None:
        self._job = None
        self._polls += 1
        if self._polls % self.GC_EVERY == 0:
            gc.collect()
        latest = self.worker.latest()
        if latest is not None:
            self.on_result(*latest)
        if not self._active:  # on_result stopped us (e.g. the face was fully captured)
            return
        if self.worker.error is not None:
            error = self.worker.error
            self.stop()
            self.on_error(error)
            return
        if self.worker.running or latest is not None:
            self._job = self.widget.after(self.interval, self._poll)
