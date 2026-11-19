"""First run: downloads the two face models (about 38 MB) with a progress bar."""
from __future__ import annotations

import queue
import threading
import tkinter as tk
from tkinter import ttk
from typing import Callable

from .. import models


class SetupView(ttk.Frame):
    """Shown instead of the main screen until the models are ready."""

    def __init__(self, parent: tk.Misc, on_ready: Callable[[], None]) -> None:
        super().__init__(parent, padding=40)
        self.on_ready = on_ready
        # The download thread can't touch widgets, so it sends messages through a queue.
        self.messages: queue.Queue = queue.Queue()

        ttk.Label(self, text="Welcome to RollCall", style="Title.TLabel").pack(pady=(40, 8))
        ttk.Label(self, text="The face recognition models are downloaded once (about 38 MB).\n"
                             "They come from OpenCV's official model zoo and are verified after downloading.",
                  justify="center", style="Muted.TLabel").pack()
        self.progress = ttk.Progressbar(self, length=420, mode="determinate", maximum=100)
        self.progress.pack(pady=(28, 8))
        self.status = ttk.Label(self, text="")
        self.status.pack()
        self.retry = ttk.Button(self, text="Try again", style="Accent.TButton", command=self.start)

    def start(self) -> None:
        self.retry.pack_forget()
        self.status.configure(text="Downloading...")
        threading.Thread(target=self._download, daemon=True).start()
        self.after(100, self._poll)

    def _download(self) -> None:
        # Runs in the background thread.
        total = sum(model.size for model in models.MODELS)
        finished = [0]  # bytes of models already completed

        def progress(model: models.Model, done: int, size: int) -> None:
            self.messages.put(("progress", (finished[0] + done) / total * 100))
            if done >= size:
                finished[0] += size

        try:
            for model in models.MODELS:
                if model.is_ready():
                    finished[0] += model.size
                else:
                    models.download(model, progress)
            self.messages.put(("done", None))
        except models.DownloadError as error:
            self.messages.put(("error", str(error)))

    def _poll(self) -> None:
        try:
            while True:
                kind, value = self.messages.get_nowait()
                if kind == "progress":
                    self.progress["value"] = value
                    self.status.configure(text=f"Downloading... {value:.0f} %")
                elif kind == "done":
                    self.on_ready()
                    return
                else:
                    self.status.configure(text=value + "\nCheck your internet connection.")
                    self.retry.pack(pady=12)
                    return
        except queue.Empty:
            pass
        self.after(100, self._poll)
