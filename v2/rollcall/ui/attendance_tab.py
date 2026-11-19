"""Take attendance: camera on the left, students present on the right."""
from __future__ import annotations

import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from typing import Optional

import numpy as np

from .. import export, paths
from ..attendance import AttendanceSession, RecognitionSettings, SessionError
from ..recognition import DEFAULT_THRESHOLD, FrameResult
from ..worker import FrameWorker
from . import theme
from .context import Context
from .video import Box, VideoCanvas, WorkerPoller


class AttendanceTab(ttk.Frame):
    def __init__(self, parent: tk.Misc, ctx: Context) -> None:
        super().__init__(parent, padding=16)
        self.ctx = ctx
        self.session: Optional[AttendanceSession] = None
        self.poller: Optional[WorkerPoller] = None
        self.last_session_id: Optional[int] = None  # the session that can be exported

        # ----- left: video and controls -----
        left = ttk.Frame(self)
        left.pack(side="left", fill="both", expand=True)
        self.video = VideoCanvas(left)
        self.video.pack(anchor="w")

        controls = ttk.Frame(left)
        controls.pack(fill="x", pady=(12, 0))
        self.start_button = ttk.Button(controls, text="Start attendance", style="Accent.TButton",
                                       command=self.start)
        self.start_button.pack(side="left")
        self.finish_button = ttk.Button(controls, text="Finish", command=self.finish, state="disabled")
        self.finish_button.pack(side="left", padx=8)
        self.export_button = ttk.Button(controls, text="Export to Excel", command=self.export,
                                        state="disabled")
        self.export_button.pack(side="left")

        # The threshold slider: how similar a face must be to count as a student.
        slider = ttk.Frame(left)
        slider.pack(fill="x", pady=(12, 0))
        ttk.Label(slider, text="Strictness").pack(side="left")
        self.threshold = tk.DoubleVar(value=DEFAULT_THRESHOLD)
        ttk.Scale(slider, from_=0.25, to=0.60, variable=self.threshold, length=220,
                  command=lambda _: self._threshold_changed()).pack(side="left", padx=8)
        self.threshold_label = ttk.Label(slider, style="Muted.TLabel")
        self.threshold_label.pack(side="left")
        self._threshold_changed()

        self.status = ttk.Label(left, text="", style="Muted.TLabel")
        self.status.pack(anchor="w", pady=(8, 0))

        # ----- right: who is present -----
        right = ttk.Frame(self, padding=(16, 0, 0, 0))
        right.pack(side="right", fill="y")
        self.count = ttk.Label(right, text="–", style="Big.TLabel")
        self.count.pack(anchor="w")
        ttk.Label(right, text="students present", style="Muted.TLabel").pack(anchor="w")
        self.present = ttk.Treeview(right, columns=("name", "time"), show="headings", height=18)
        self.present.heading("name", text="Name")
        self.present.heading("time", text="Time")
        self.present.column("name", width=200)
        self.present.column("time", width=70, anchor="center")
        self.present.pack(fill="y", expand=True, pady=(10, 0))

        self.after_idle(self.check_unfinished)

    # ---------- an unfinished session from last time ----------

    def check_unfinished(self) -> None:
        """If the app closed during a session, offer to continue it."""
        open_session = self.ctx.db.open_session()
        if open_session is None:
            return
        answer = messagebox.askyesnocancel(
            "Unfinished session",
            f"The session started at {open_session.started_at} was not finished.\n\n"
            "Yes: continue taking attendance\nNo: finish it now\nCancel: decide later",
            parent=self)
        if answer is True:
            self.start(resume=open_session.id)
        elif answer is False:
            self.ctx.db.end_session(open_session.id)
            self.last_session_id = open_session.id
            self.export_button.configure(state="normal")
            self.ctx.changed()

    # ---------- running ----------

    def start(self, resume: Optional[int] = None) -> None:
        if self.ctx.camera_in_use:
            messagebox.showinfo("Camera busy", "Close the registration window first.", parent=self)
            return
        settings = RecognitionSettings(threshold=self.threshold.get())
        try:
            if resume is None:
                self.session = AttendanceSession.start(self.ctx.db, self.ctx.engine, settings)
            else:
                self.session = AttendanceSession.resume(self.ctx.db, self.ctx.engine, resume, settings)
        except SessionError as error:
            messagebox.showwarning("Can't start", str(error), parent=self)
            return

        self.ctx.camera_in_use = True
        self.video.message("Starting camera...")
        worker = FrameWorker(self.ctx.open_camera, self.session.analyze)
        self.poller = WorkerPoller(self, worker, self._on_frame, self._on_camera_error).start()
        self.start_button.configure(state="disabled")
        self.finish_button.configure(state="normal")
        self.export_button.configure(state="disabled")
        self.status.configure(text="Students are marked present after being recognized for a moment.")
        self._refresh_present()
        self.ctx.changed()  # the new open session appears in History

    def _on_frame(self, frame: np.ndarray, result: FrameResult) -> None:
        if self.session is None:
            return
        saved = self.session.record()  # saves every student confirmed since last time (in this thread)
        self.video.show(frame, [self._box(face) for face in result.faces])
        if saved:
            self._refresh_present()

    def _box(self, labeled) -> Box:
        x, y, w, h = labeled.face.box
        if labeled.student_id is None:
            return Box(x, y, w, h, theme.UNKNOWN, "Unknown")
        name = self.session.student(labeled.student_id).full_name
        if labeled.confirmed:
            return Box(x, y, w, h, theme.PRESENT, f"✓ {name}")
        return Box(x, y, w, h, theme.PENDING, f"{name} {labeled.progress:.0%}")

    def _on_camera_error(self, error: BaseException) -> None:
        self.video.message("Camera error")
        messagebox.showerror("Camera", f"{error}\n\nThe students already marked are saved. "
                                       "Press Start to continue the session.", parent=self)
        # Keep the session open so it can be resumed; release the camera.
        session_id = self.session.session_id if self.session else None
        self._stop_camera()
        self.session = None
        if session_id is not None:
            self.start_button.configure(text="Continue session", state="normal",
                                        command=lambda: self._continue(session_id))
        else:
            self.start_button.configure(state="normal")
        self.finish_button.configure(state="disabled")

    def _continue(self, session_id: int) -> None:
        self.start_button.configure(text="Start attendance", command=self.start)
        self.start(resume=session_id)

    def finish(self) -> None:
        if self.session is None:
            return
        self._stop_camera()
        finished = self.session.finish()
        self.last_session_id = finished.id
        self.session = None
        self.video.message(f"Session finished: {finished.present} of {finished.enrolled} present")
        self.start_button.configure(state="normal")
        self.finish_button.configure(state="disabled")
        self.export_button.configure(state="normal")
        self.status.configure(text="")
        self.ctx.changed()

    def _stop_camera(self) -> None:
        if self.poller is not None:
            self.poller.stop()
            self.poller = None
        self.ctx.camera_in_use = False

    @property
    def running(self) -> bool:
        return self.session is not None

    def shutdown(self) -> None:
        """Called when the window closes during a session."""
        self._stop_camera()

    # ---------- helpers ----------

    def _threshold_changed(self) -> None:
        value = self.threshold.get()
        hint = "default" if abs(value - DEFAULT_THRESHOLD) < 0.005 else (
            "stricter" if value > DEFAULT_THRESHOLD else "more lenient")
        self.threshold_label.configure(text=f"{value:.2f} ({hint})")
        if self.session is not None:
            self.session.threshold = value

    def _refresh_present(self) -> None:
        session_id = self.session.session_id if self.session else self.last_session_id
        if session_id is None:
            return
        report = self.ctx.db.session_report(session_id)
        present = [row for row in report if row.present]
        self.count.configure(text=f"{len(present)} / {len(report)}")
        self.present.delete(*self.present.get_children())
        for row in sorted(present, key=lambda r: r.seen_at, reverse=True):  # newest on top
            self.present.insert("", "end", values=(row.student.full_name, row.seen_at[11:16]))

    def export(self) -> None:
        if self.last_session_id is None:
            return
        save_session(self, self.ctx, self.last_session_id)


def save_session(parent: tk.Misc, ctx: Context, session_id: int) -> None:
    """Asks where to save one session and writes the Excel file."""
    path = filedialog.asksaveasfilename(
        parent=parent, title="Export attendance", defaultextension=".xlsx",
        initialdir=paths.exports_dir(), initialfile=export.session_filename(ctx.db, session_id),
        filetypes=[("Excel workbook", "*.xlsx")])
    if not path:
        return  # the user cancelled
    try:
        export.session_workbook(ctx.db, session_id).save(path)
    except OSError as error:  # no permission, file open in Excel, disk full...
        messagebox.showerror("Export", f"Couldn't save the file:\n{error}", parent=parent)
        return
    messagebox.showinfo("Export", f"Saved:\n{path}", parent=parent)
