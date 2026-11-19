"""Students: search, add, edit names, retake face, delete."""
from __future__ import annotations

import tkinter as tk
from tkinter import messagebox, ttk
from typing import Optional

from ..database import Student
from . import theme
from .context import Context
from .register_dialog import RegisterDialog

COLUMNS = ("id", "last", "first", "faces", "attendance")
HEADINGS = ("Account number", "Last names", "First names", "Face samples", "Attendance")


class StudentsTab(ttk.Frame):
    def __init__(self, parent: tk.Misc, ctx: Context) -> None:
        super().__init__(parent, padding=16)
        self.ctx = ctx

        top = ttk.Frame(self)
        top.pack(fill="x")
        ttk.Label(top, text="Search").pack(side="left")
        self.search = tk.StringVar()
        self.search.trace_add("write", lambda *_: self.refresh())  # filter while typing
        ttk.Entry(top, textvariable=self.search, width=30).pack(side="left", padx=8)
        ttk.Button(top, text="Add student", style="Accent.TButton", command=self.add).pack(side="right")

        self.table = ttk.Treeview(self, columns=COLUMNS, show="headings", selectmode="browse")
        for column, heading in zip(COLUMNS, HEADINGS):
            self.table.heading(column, text=heading)
        self.table.column("id", width=130)
        self.table.column("faces", width=110, anchor="center")
        self.table.column("attendance", width=110, anchor="center")
        self.table.tag_configure("no_face", foreground=theme.UNKNOWN)
        self.table.pack(fill="both", expand=True, pady=10)
        self.table.bind("<<TreeviewSelect>>", lambda _: self._update_buttons())
        self.table.bind("<Double-1>", lambda _: self.edit())

        actions = ttk.Frame(self)
        actions.pack(fill="x")
        self.edit_button = ttk.Button(actions, text="Edit names", command=self.edit)
        self.retake_button = ttk.Button(actions, text="Retake face", command=self.retake)
        self.delete_button = ttk.Button(actions, text="Delete", command=self.delete)
        for button in (self.edit_button, self.retake_button, self.delete_button):
            button.pack(side="left", padx=(0, 8))
        self.total = ttk.Label(actions, style="Muted.TLabel")
        self.total.pack(side="right")

        ctx.on_change(self.refresh)
        self.refresh()

    def refresh(self) -> None:
        selected = self.selected()
        stats = {s.student.id: s for s in self.ctx.db.student_stats()}
        students = self.ctx.db.list_students(self.search.get())
        self.table.delete(*self.table.get_children())
        for student in students:
            faces = self.ctx.db.embedding_count(student.id)
            pct = stats[student.id].percentage
            self.table.insert("", "end", iid=str(student.id), tags=() if faces else ("no_face",), values=(
                student.id, student.last_names, student.first_names,
                faces if faces else "none",
                "—" if pct is None else f"{pct:.0f} %  ({stats[student.id].attended}/{stats[student.id].total})",
            ))
        if selected is not None and self.table.exists(str(selected.id)):
            self.table.selection_set(str(selected.id))
        self.total.configure(text=f"{len(students)} student(s)")
        self._update_buttons()

    def selected(self) -> Optional[Student]:
        selection = self.table.selection()
        return self.ctx.db.get_student(int(selection[0])) if selection else None

    def _update_buttons(self) -> None:
        state = "normal" if self.table.selection() else "disabled"
        for button in (self.edit_button, self.retake_button, self.delete_button):
            button.configure(state=state)

    # ---------- actions ----------

    def add(self) -> None:
        RegisterDialog(self, self.ctx, on_saved=self._saved)

    def retake(self) -> None:
        student = self.selected()
        if student:
            RegisterDialog(self, self.ctx, on_saved=self._saved, retake=student)

    def _saved(self, student: Student) -> None:
        self.ctx.changed()
        self.table.selection_set(str(student.id))
        self.table.see(str(student.id))

    def edit(self) -> None:
        student = self.selected()
        if student:
            EditDialog(self, self.ctx, student)

    def delete(self) -> None:
        student = self.selected()
        if student is None:
            return
        sure = messagebox.askyesno(
            "Delete student",
            f"Delete {student.full_name} ({student.id})?\n\n"
            "Their face data and attendance history are deleted too. This can't be undone.",
            icon="warning", parent=self)
        if sure:
            self.ctx.db.delete_student(student.id)
            self.ctx.changed()


class EditDialog(tk.Toplevel):
    """Changes a student's names. The account number can't change: it identifies them."""

    def __init__(self, parent: tk.Misc, ctx: Context, student: Student) -> None:
        super().__init__(parent)
        self.ctx, self.student = ctx, student
        self.title("Edit student")
        self.configure(bg=theme.BG)
        self.resizable(False, False)
        self.transient(parent)

        body = ttk.Frame(self, padding=16)
        body.pack()
        ttk.Label(body, text=f"Account number: {student.id}", style="Muted.TLabel").grid(
            row=0, column=0, columnspan=2, sticky="w", pady=(0, 8))
        self.first = tk.StringVar(value=student.first_names)
        self.last = tk.StringVar(value=student.last_names)
        for row, (label, var) in enumerate([("First name(s)", self.first), ("Last name(s)", self.last)], 1):
            ttk.Label(body, text=label).grid(row=row, column=0, sticky="w", pady=4)
            ttk.Entry(body, textvariable=var, width=32).grid(row=row, column=1, padx=(10, 0), pady=4)
        self.error = ttk.Label(body, text="", foreground=theme.UNKNOWN)
        self.error.grid(row=3, column=0, columnspan=2, sticky="w")
        buttons = ttk.Frame(body)
        buttons.grid(row=4, column=0, columnspan=2, sticky="e", pady=(10, 0))
        ttk.Button(buttons, text="Cancel", command=self.destroy).pack(side="right")
        ttk.Button(buttons, text="Save", style="Accent.TButton", command=self.save).pack(side="right", padx=8)
        self.bind("<Return>", lambda _: self.save())
        self.bind("<Escape>", lambda _: self.destroy())
        self.grab_set()

    def save(self) -> None:
        try:
            self.ctx.db.update_student(self.student.id, self.first.get(), self.last.get())
        except ValueError as error:
            self.error.configure(text=str(error))
            return
        self.destroy()
        self.ctx.changed()
