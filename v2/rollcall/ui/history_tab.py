"""History: past sessions on the left, the selected session's list on the right."""
from __future__ import annotations

import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from typing import Optional

from .. import export, paths
from . import theme
from .attendance_tab import save_session
from .context import Context


class HistoryTab(ttk.Frame):
    def __init__(self, parent: tk.Misc, ctx: Context) -> None:
        super().__init__(parent, padding=16)
        self.ctx = ctx

        # ----- left: sessions -----
        left = ttk.Frame(self)
        left.pack(side="left", fill="both", expand=True)
        ttk.Label(left, text="Sessions", style="Title.TLabel").pack(anchor="w")
        self.sessions = ttk.Treeview(left, columns=("date", "time", "present", "pct"),
                                     show="headings", selectmode="browse")
        for column, heading, width in [("date", "Date", 100), ("time", "Time", 110),
                                       ("present", "Present", 80), ("pct", "Attendance", 90)]:
            self.sessions.heading(column, text=heading)
            self.sessions.column(column, width=width, anchor="center")
        self.sessions.tag_configure("open", foreground=theme.PENDING)
        self.sessions.pack(fill="both", expand=True, pady=(8, 8))
        self.sessions.bind("<<TreeviewSelect>>", lambda _: self.show_selected())

        buttons = ttk.Frame(left)
        buttons.pack(fill="x")
        self.export_button = ttk.Button(buttons, text="Export session", command=self.export_session)
        self.export_button.pack(side="left")
        self.delete_button = ttk.Button(buttons, text="Delete session", command=self.delete_session)
        self.delete_button.pack(side="left", padx=8)
        ttk.Button(buttons, text="Export full history", style="Accent.TButton",
                   command=self.export_history).pack(side="right")

        # ----- right: the selected session -----
        right = ttk.Frame(self, padding=(16, 0, 0, 0))
        right.pack(side="right", fill="both", expand=True)
        self.detail_title = ttk.Label(right, text="Select a session", style="Title.TLabel")
        self.detail_title.pack(anchor="w")
        self.detail = ttk.Treeview(right, columns=("status", "name", "seen"), show="headings")
        for column, heading, width in [("status", "Status", 80), ("name", "Name", 220), ("seen", "Seen at", 80)]:
            self.detail.heading(column, text=heading)
            self.detail.column(column, width=width, anchor="w" if column == "name" else "center")
        self.detail.tag_configure("absent", foreground=theme.UNKNOWN)
        self.detail.pack(fill="both", expand=True, pady=(8, 0))

        ctx.on_change(self.refresh)
        self.refresh()

    def refresh(self) -> None:
        selected = self.selected_id()
        self.sessions.delete(*self.sessions.get_children())
        for s in self.ctx.db.list_sessions():
            date, start = s.started_at[:10], s.started_at[11:16]
            end = s.ended_at[11:16] if s.ended_at else "now"
            pct = f"{s.present / s.enrolled:.0%}" if s.enrolled else "—"
            self.sessions.insert("", "end", iid=str(s.id), tags=("open",) if s.is_open else (),
                                 values=(date, f"{start}–{end}", f"{s.present}/{s.enrolled}", pct))
        if selected is not None and self.sessions.exists(str(selected)):
            self.sessions.selection_set(str(selected))
        self.show_selected()

    def selected_id(self) -> Optional[int]:
        selection = self.sessions.selection()
        return int(selection[0]) if selection else None

    def show_selected(self) -> None:
        session_id = self.selected_id()
        self.detail.delete(*self.detail.get_children())
        state = "normal" if session_id is not None else "disabled"
        self.export_button.configure(state=state)
        self.delete_button.configure(state=state)
        if session_id is None:
            self.detail_title.configure(text="Select a session")
            return
        session = self.ctx.db.get_session(session_id)
        self.detail_title.configure(text=f"{session.started_at[:16]}  ·  {session.present} of {session.enrolled} present")
        for row in self.ctx.db.session_report(session_id):
            self.detail.insert("", "end", tags=() if row.present else ("absent",), values=(
                "Present" if row.present else "Absent", row.student.full_name,
                row.seen_at[11:16] if row.seen_at else ""))

    # ---------- actions ----------

    def export_session(self) -> None:
        session_id = self.selected_id()
        if session_id is not None:
            save_session(self, self.ctx, session_id)

    def export_history(self) -> None:
        path = filedialog.asksaveasfilename(
            parent=self, title="Export history", defaultextension=".xlsx",
            initialdir=paths.exports_dir(), initialfile=export.history_filename(),
            filetypes=[("Excel workbook", "*.xlsx")])
        if not path:
            return
        try:
            export.history_workbook(self.ctx.db).save(path)
        except OSError as error:
            messagebox.showerror("Export", f"Couldn't save the file:\n{error}", parent=self)
            return
        messagebox.showinfo("Export", f"Saved:\n{path}", parent=self)

    def delete_session(self) -> None:
        session_id = self.selected_id()
        if session_id is None:
            return
        session = self.ctx.db.get_session(session_id)
        if session.is_open:
            messagebox.showinfo("Delete session", "This session is still running. Finish it first.", parent=self)
            return
        if messagebox.askyesno("Delete session",
                               f"Delete the session of {session.started_at[:16]} and its attendance?\n"
                               "This can't be undone.", icon="warning", parent=self):
            self.ctx.db.delete_session(session_id)
            self.ctx.changed()
