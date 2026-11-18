"""What goes in each Excel file.

    session_workbook(db, id)  one session: present and absent students
    history_workbook(db)      every session: totals per student, per session,
                              and a grid of who came to which session

Building a workbook and saving it are separate steps: the *_workbook functions
only read the database and return a Workbook (easy to test), and the export_*
functions save it with a name that never overwrites an earlier export.
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Optional

from . import paths
from .database import TIME_FORMAT, Database
from .xlsx import Percent, Workbook

PRESENT, ABSENT = "Present", "Absent"


def session_workbook(db: Database, session_id: int) -> Workbook:
    session = db.get_session(session_id)
    if session is None:
        raise KeyError(f"No session {session_id}.")
    report = db.session_report(session_id)

    book = Workbook()
    sheet = book.add_sheet("Attendance")
    sheet.append(["Attendance"], bold=True)
    sheet.append(["Started", parse_time(session.started_at)])
    sheet.append(["Finished", parse_time(session.ended_at) if session.ended_at else "In progress"])
    sheet.append(["Present", f"{session.present} of {session.enrolled}", _ratio(session.present, session.enrolled)])
    sheet.append()
    sheet.append(["Account number", "Last names", "First names", "Status", "Time seen", "Similarity"], bold=True)
    for row in report:
        sheet.append([
            row.student.id,
            row.student.last_names,
            row.student.first_names,
            PRESENT if row.present else ABSENT,
            parse_time(row.seen_at) if row.seen_at else None,
            round(row.similarity, 3) if row.similarity is not None else None,
        ])
    sheet.set_widths(16, 22, 22, 10, 18, 11)
    sheet.freeze(6)  # the header stays visible while scrolling the list
    return book


def history_workbook(db: Database) -> Workbook:
    book = Workbook()

    # Sheet 1: one row per student, with their attendance percentage.
    students = book.add_sheet("Students")
    students.append(["Account number", "Last names", "First names", "Attended", "Sessions", "Attendance"], bold=True)
    for stats in db.student_stats():
        s = stats.student
        students.append([s.id, s.last_names, s.first_names, stats.attended, stats.total,
                         _ratio(stats.attended, stats.total)])
    students.set_widths(16, 22, 22, 10, 10, 12)
    students.freeze(1)

    # Sheet 2: one row per session, oldest first.
    sessions = list(reversed(db.list_sessions()))  # the database returns newest first
    summary = book.add_sheet("Sessions")
    summary.append(["Session", "Started", "Finished", "Present", "Enrolled", "Attendance"], bold=True)
    for session in sessions:
        summary.append([session.id, parse_time(session.started_at),
                        parse_time(session.ended_at) if session.ended_at else "In progress",
                        session.present, session.enrolled, _ratio(session.present, session.enrolled)])
    summary.set_widths(10, 18, 18, 10, 10, 12)
    summary.freeze(1)

    # Sheet 3: students x sessions. "P" present, "A" absent, empty = not registered yet.
    grid = book.add_sheet("By session")
    grid.append(["Account number", "Last names", "First names"]
                + [parse_time(s.started_at) for s in sessions], bold=True)
    marks = {s.id: {row.student.id: ("P" if row.present else "A") for row in db.session_report(s.id)}
             for s in sessions}
    for student in db.list_students():
        grid.append([student.id, student.last_names, student.first_names]
                    + [marks[s.id].get(student.id) for s in sessions])
    grid.set_widths(16, 22, 22, *[17] * len(sessions))
    grid.freeze(1)
    return book


def session_filename(db: Database, session_id: int) -> str:
    """Attendance_<date>_<time>.xlsx, from when the session started."""
    session = db.get_session(session_id)
    if session is None:
        raise KeyError(f"No session {session_id}.")
    return f"Attendance_{parse_time(session.started_at).strftime('%Y-%m-%d_%H-%M')}.xlsx"


def history_filename(today: Optional[datetime] = None) -> str:
    return f"Attendance_history_{(today or datetime.now()).strftime('%Y-%m-%d')}.xlsx"


def export_session(db: Database, session_id: int, folder: Optional[Path] = None) -> Path:
    """Saves one session in `folder` (Downloads by default). Returns the file's path."""
    target = unique_path(Path(folder or paths.exports_dir()) / session_filename(db, session_id))
    return session_workbook(db, session_id).save(target)


def export_history(db: Database, folder: Optional[Path] = None, today: Optional[datetime] = None) -> Path:
    """Saves every session in `folder` (Downloads by default). Returns the file's path."""
    target = unique_path(Path(folder or paths.exports_dir()) / history_filename(today))
    return history_workbook(db).save(target)


# ---------- helpers ----------

def parse_time(text: str) -> datetime:
    """A timestamp from the database as a datetime, so Excel shows it as a real date."""
    return datetime.strptime(text, TIME_FORMAT)


def unique_path(path: Path) -> Path:
    """The same path, or "name (2).xlsx", "name (3).xlsx"... if it already exists."""
    candidate, number = path, 2
    while candidate.exists():
        candidate = path.with_name(f"{path.stem} ({number}){path.suffix}")
        number += 1
    return candidate


def _ratio(part: int, total: int) -> Optional[Percent]:
    # Empty cell instead of 0 % when there's nothing to count: "no data" is not "nobody came".
    return Percent(part / total) if total else None
