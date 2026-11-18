"""The window to register a new student, or to retake a student's face.

    new student:  type the data -> "Continue" -> camera collects the face -> saved
    retake face:  camera collects the face -> replaces the old embeddings
"""
from __future__ import annotations

import tkinter as tk
from tkinter import messagebox, ttk
from typing import Callable, Optional

import numpy as np

from ..database import Student
from ..recognition import EnrollStatus, EnrollStep
from ..registration import Registration
from ..worker import FrameWorker
from . import theme
from .context import Context
from .video import Box, VideoCanvas, WorkerPoller


class RegisterDialog(tk.Toplevel):
    def __init__(self, parent: tk.Misc, ctx: Context, on_saved: Callable[[Student], None],
                 retake: Optional[Student] = None) -> None:
        super().__init__(parent)
        self.ctx, self.on_saved, self.retake = ctx, on_saved, retake
        self.registration: Optional[Registration] = None
        self.poller: Optional[WorkerPoller] = None
        self.data: tuple = ()  # the validated (id, first names, last names)

        self.title("Retake face" if retake else "Add student")
        self.configure(bg=theme.BG)
        self.resizable(False, False)
        self.transient(parent)  # stays on top of the main window
        self.protocol("WM_DELETE_WINDOW", self.cancel)

        body = ttk.Frame(self, padding=16)
        body.pack(fill="both", expand=True)

        # ----- the form (only for new students) -----
        self.form = ttk.Frame(body)
        self.account = tk.StringVar()
        self.first = tk.StringVar()
        self.last = tk.StringVar()
        for row, (label, var) in enumerate([("Account number", self.account),
                                            ("First name(s)", self.first),
                                            ("Last name(s)", self.last)]):
            ttk.Label(self.form, text=label).grid(row=row, column=0, sticky="w", pady=4)
            entry = ttk.Entry(self.form, textvariable=var, width=32)
            entry.grid(row=row, column=1, pady=4, padx=(10, 0))
            if row == 0:
                entry.focus_set()
        self.form_error = ttk.Label(self.form, text="", foreground=theme.UNKNOWN)
        self.form_error.grid(row=3, column=0, columnspan=2, sticky="w", pady=(6, 0))
        ttk.Button(self.form, text="Continue to camera", style="Accent.TButton",
                   command=self.continue_to_camera).grid(row=4, column=1, sticky="e", pady=(10, 0))
        self.bind("<Return>", lambda _: self.continue_to_camera() if self.form.winfo_ismapped() else None)

        # ----- the camera step -----
        self.camera_step = ttk.Frame(body)
        self.who = ttk.Label(self.camera_step, style="Title.TLabel")
        self.who.pack(anchor="w")
        self.video = VideoCanvas(self.camera_step, 560, 420)
        self.video.pack(pady=(8, 8))
        self.hint = ttk.Label(self.camera_step, text="", style="Hint.TLabel")
        self.hint.pack(anchor="w")
        self.progress = ttk.Progressbar(self.camera_step, length=560, mode="determinate")
        self.progress.pack(pady=(6, 0))
        ttk.Button(self.camera_step, text="Cancel", command=self.cancel).pack(anchor="e", pady=(10, 0))

        if retake:
            self.data = (retake.id, retake.first_names, retake.last_names)
            self.start_camera()
        else:
            self.form.pack(fill="both")
        self.grab_set()  # modal: the main window waits until this one closes

    # ---------- step 1: the form ----------

    def continue_to_camera(self) -> None:
        try:
            self.data = Registration.check(self.ctx.db, self.account.get(), self.first.get(), self.last.get())
        except ValueError as error:  # includes DuplicateStudentError
            self.form_error.configure(text=str(error))
            return
        self.form.pack_forget()
        self.start_camera()

    # ---------- step 2: the camera ----------

    def start_camera(self) -> None:
        if self.ctx.camera_in_use:
            messagebox.showinfo("Camera busy", "Finish taking attendance first.", parent=self)
            self.destroy()
            return
        self.registration = Registration(self.ctx.db, self.ctx.engine,
                                         retake=self.retake.id if self.retake else None)
        self.ctx.camera_in_use = True
        student_id, first, last = self.data
        self.who.configure(text=f"{first} {last}  ·  {student_id}")
        self.camera_step.pack(fill="both")
        self.progress.configure(maximum=self.registration.enroller.needed, value=0)
        self.hint.configure(text="Starting camera...")
        worker = FrameWorker(self.ctx.open_camera, self.registration.analyze)
        self.poller = WorkerPoller(self, worker, self._on_frame, self._on_error).start()

    def _on_frame(self, frame: np.ndarray, step: EnrollStep) -> None:
        boxes = []
        if step.face is not None:
            ok = step.status in (EnrollStatus.CAPTURED, EnrollStatus.WAITING, EnrollStatus.DONE)
            boxes.append(Box(*step.face.box, theme.PRESENT if ok else theme.PENDING))
        self.video.show(frame, boxes)
        self.progress.configure(value=step.captured)
        self.hint.configure(text=f"{step.status.value}   ({step.captured}/{step.needed})")
        if step.status == EnrollStatus.DONE:
            self._stop_camera()
            self.save()

    def _on_error(self, error: BaseException) -> None:
        self._stop_camera()
        messagebox.showerror("Camera", str(error), parent=self)
        self.destroy()

    # ---------- step 3: saving ----------

    def save(self) -> None:
        duplicate = self.registration.duplicate()
        if duplicate is not None:
            other, similarity = duplicate
            keep = messagebox.askyesno(
                "This face is already registered?",
                f"This face looks like {other.full_name} ({other.id}), similarity {similarity:.2f}.\n\n"
                "Save it anyway? Choose No if it's the same person registered twice.",
                icon="warning", parent=self)
            if not keep:
                self.destroy()
                return
        try:
            student = self.registration.save(*self.data)
        except ValueError as error:  # e.g. someone registered the same number meanwhile
            messagebox.showerror("Couldn't save", str(error), parent=self)
            self.destroy()
            return
        self.destroy()
        self.on_saved(student)

    def cancel(self) -> None:
        self._stop_camera()
        self.destroy()

    def _stop_camera(self) -> None:
        if self.poller is not None:
            self.poller.stop()
            self.poller = None
            self.ctx.camera_in_use = False
