from datetime import datetime

import pytest

from rollcall import export
from rollcall.xlsx import excel_serial
import xlsx_reader


def serial(text):
    return pytest.approx(excel_serial(datetime.strptime(text, "%Y-%m-%d %H:%M:%S")), abs=1e-6)


@pytest.fixture
def history(db, clock):
    """Ana and Beto from the start, Carla joins after the first session.

    Session 1 (Jan 13): Ana present, Beto absent
    Session 2 (Jan 14): Ana and Carla present, Beto absent
    """
    db.add_student(1, "Ana", "Lopez")
    db.add_student(2, "Beto", "Diaz")
    clock.advance(days=1)
    first = db.start_session()
    clock.advance(minutes=3)
    db.mark_present(first, 1, 0.81234)
    clock.advance(minutes=47)
    db.end_session(first)

    clock.advance(hours=1)
    db.add_student(3, "Carla", "Ruiz")
    clock.advance(days=1)
    second = db.start_session()
    db.mark_present(second, 1, 0.7)
    db.mark_present(second, 3, 0.9)
    db.end_session(second)
    return db, first, second


def test_session_workbook(history, tmp_path):
    db, first, _ = history
    path = export.session_workbook(db, first).save(tmp_path / "s.xlsx")
    rows = xlsx_reader.values(path, "Attendance")

    assert rows[0] == ["Attendance"]
    assert rows[1] == ["Started", serial("2026-01-13 08:00:00")]
    assert rows[2] == ["Finished", serial("2026-01-13 08:50:00")]
    assert rows[3] == ["Present", "1 of 2", pytest.approx(0.5)]
    assert rows[5] == ["Account number", "Last names", "First names", "Status", "Time seen", "Similarity"]
    assert rows[6] == [1, "Lopez", "Ana", "Present", serial("2026-01-13 08:03:00"), pytest.approx(0.812)]
    assert rows[7] == [2, "Diaz", "Beto", "Absent"]  # time and similarity left empty
    assert len(rows) == 8                            # Carla wasn't registered yet: not absent


def test_open_session_says_in_progress(db, tmp_path):
    db.add_student(1, "Ana", "Lopez")
    session = db.start_session()
    rows = xlsx_reader.values(export.session_workbook(db, session).save(tmp_path / "o.xlsx"), "Attendance")
    assert rows[2] == ["Finished", "In progress"]


def test_history_workbook(history, tmp_path):
    db, first, second = history
    path = export.history_workbook(db).save(tmp_path / "h.xlsx")

    students = xlsx_reader.values(path, "Students")
    assert students[0] == ["Account number", "Last names", "First names", "Attended", "Sessions", "Attendance"]
    assert students[1:] == [
        [2, "Diaz", "Beto", 0, 2, 0.0],                      # came to none of 2
        [1, "Lopez", "Ana", 2, 2, 1.0],                      # 100 %
        [3, "Ruiz", "Carla", 1, 1, 1.0],                     # only 1 session since she joined
    ]

    sessions = xlsx_reader.values(path, "Sessions")
    assert [row[0] for row in sessions[1:]] == [first, second]  # oldest first
    assert sessions[1][3:] == [1, 2, pytest.approx(0.5)]
    assert sessions[2][3:] == [2, 3, pytest.approx(2 / 3)]

    grid = xlsx_reader.values(path, "By session")
    assert grid[0][3:] == [serial("2026-01-13 08:00:00"), serial("2026-01-14 09:50:00")]
    assert grid[1:] == [
        [2, "Diaz", "Beto", "A", "A"],
        [1, "Lopez", "Ana", "P", "P"],
        [3, "Ruiz", "Carla", None, "P"],                     # empty: not registered for session 1
    ]


def test_history_with_no_sessions_or_students(db, tmp_path):
    path = export.history_workbook(db).save(tmp_path / "empty.xlsx")
    assert len(xlsx_reader.values(path, "Students")) == 1   # just the header
    assert len(xlsx_reader.values(path, "By session")) == 1


def test_student_without_sessions_has_an_empty_percentage(db, tmp_path):
    db.add_student(1, "Ana", "Lopez")
    rows = xlsx_reader.values(export.history_workbook(db).save(tmp_path / "x.xlsx"), "Students")
    assert rows[1] == [1, "Lopez", "Ana", 0, 0]  # no "0 %": there's nothing to count yet


def test_export_files_never_overwrite_each_other(history, tmp_path):
    db, first, _ = history
    one = export.export_session(db, first, tmp_path)
    two = export.export_session(db, first, tmp_path)
    assert one.name == "Attendance_2026-01-13_08-00.xlsx"
    assert two.name == "Attendance_2026-01-13_08-00 (2).xlsx"

    history_file = export.export_history(db, tmp_path, today=datetime(2026, 2, 1))
    assert history_file.name == "Attendance_history_2026-02-01.xlsx"


def test_exports_default_to_the_downloads_folder(history, tmp_path, monkeypatch):
    db, first, _ = history
    monkeypatch.setattr(export.paths, "exports_dir", lambda: tmp_path)
    assert export.export_session(db, first).parent == tmp_path


def test_missing_session(db, tmp_path):
    with pytest.raises(KeyError):
        export.export_session(db, 42, tmp_path)
