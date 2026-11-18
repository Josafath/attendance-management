"""The main window. Builds everything and wires it together.

Nothing here creates its own dependencies: the database, the camera and the
face engine are passed in (dependency injection). main() passes the real ones;
the UI tests pass a fake camera and a fake engine.
"""
from __future__ import annotations

import os
import sys
import tkinter as tk
from tkinter import messagebox, ttk
from typing import Callable, Optional

from .. import __version__, models, paths
from ..camera import Camera, Webcam
from ..database import Database
from ..faces import FaceEngine, OpenCVFaceEngine
from . import theme
from .attendance_tab import AttendanceTab
from .context import Context
from .history_tab import HistoryTab
from .setup import SetupView
from .students_tab import StudentsTab


class App(tk.Tk):
    def __init__(self, db: Database, open_camera: Callable[[], Camera],
                 make_engine: Callable[[], FaceEngine],
                 needs_setup: Callable[[], bool] = lambda: not models.models_ready()) -> None:
        super().__init__()
        self.db, self.open_camera, self.make_engine = db, open_camera, make_engine
        self.ctx: Optional[Context] = None
        self.attendance: Optional[AttendanceTab] = None

        self.title(f"RollCall {__version__}")
        self.minsize(1000, 640)
        theme.apply(self)
        self.protocol("WM_DELETE_WINDOW", self.close)

        if needs_setup():
            self.setup = SetupView(self, on_ready=self._ready)
            self.setup.pack(fill="both", expand=True)
            self.setup.start()
        else:
            self._ready()

    def _ready(self) -> None:
        """The models are there: build the real screen."""
        if getattr(self, "setup", None) is not None:
            self.setup.destroy()
            self.setup = None
        self.ctx = Context(self.db, self.make_engine(), self.open_camera)

        self.tabs = ttk.Notebook(self)
        self.tabs.pack(fill="both", expand=True, padx=8, pady=8)
        self.attendance = AttendanceTab(self.tabs, self.ctx)
        self.students = StudentsTab(self.tabs, self.ctx)
        self.history = HistoryTab(self.tabs, self.ctx)
        self.tabs.add(self.attendance, text="Take attendance")
        self.tabs.add(self.students, text="Students")
        self.tabs.add(self.history, text="History")
        # Lists are refreshed when their tab is shown, so they're never out of date.
        self.tabs.bind("<<NotebookTabChanged>>", lambda _: self._refresh_visible_tab())

    def _refresh_visible_tab(self) -> None:
        current = self.nametowidget(self.tabs.select())
        if current in (self.students, self.history):
            current.refresh()

    def close(self) -> None:
        """Closing during a session: finish it, keep it open for later, or stay."""
        if self.attendance is not None and self.attendance.running:
            answer = messagebox.askyesnocancel(
                "Session in progress",
                "Finish the attendance session before closing?\n\n"
                "Yes: finish it\nNo: keep it open and continue it next time\nCancel: don't close",
                parent=self)
            if answer is None:
                return
            if answer:
                self.attendance.finish()
            else:
                self.attendance.shutdown()
        self.destroy()


def main() -> None:
    """Entry point: `rollcall` or `python -m rollcall`."""
    # ROLLCALL_CAMERA picks another camera (0 is the default one; an external webcam is often 1).
    camera_index = int(os.environ.get("ROLLCALL_CAMERA", "0"))
    try:
        db = Database(paths.database_file())
    except Exception as error:  # e.g. the file belongs to a newer version
        print(f"RollCall can't open its database ({paths.database_file()}):\n{error}", file=sys.stderr)
        sys.exit(1)
    try:
        App(db, open_camera=lambda: Webcam(camera_index), make_engine=OpenCVFaceEngine).mainloop()
    finally:
        db.close()
