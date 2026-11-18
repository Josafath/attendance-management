import sqlite3

import numpy as np
import pytest

from rollcall.database import Database, DuplicateStudentError, SCHEMA_VERSION
from conftest import random_embedding


# ---------- students ----------

def test_add_and_get_student(db):
    student = db.add_student(101, "  Ana   Maria ", "Lopez")
    assert student.id == 101
    assert student.first_names == "Ana Maria"  # extra spaces removed
    assert student.full_name == "Ana Maria Lopez"
    assert student.created_at == "2026-01-12 08:00:00"
    assert db.get_student(101) == student
    assert db.get_student(999) is None


@pytest.mark.parametrize("student_id", [0, -5, "101", 1.5, True, None])
def test_account_number_must_be_a_positive_int(db, student_id):
    with pytest.raises(ValueError):
        db.add_student(student_id, "Ana", "Lopez")


@pytest.mark.parametrize("first, last", [("", "Lopez"), ("Ana", "   "), ("  ", "")])
def test_names_are_required(db, first, last):
    with pytest.raises(ValueError):
        db.add_student(101, first, last)


def test_duplicate_account_number_is_rejected(db):
    db.add_student(101, "Ana", "Lopez")
    with pytest.raises(DuplicateStudentError):
        db.add_student(101, "Other", "Person")
    assert db.get_student(101).first_names == "Ana"  # the original is untouched


def test_update_student(db):
    db.add_student(101, "Ana", "Lopez")
    updated = db.update_student(101, "Ana Sofia", "Lopez Ruiz")
    assert updated.full_name == "Ana Sofia Lopez Ruiz"
    with pytest.raises(KeyError):
        db.update_student(999, "X", "Y")


def test_list_students_sorted_by_last_names(db):
    db.add_student(3, "Carla", "Ruiz")
    db.add_student(1, "Ana", "Lopez")
    db.add_student(2, "Beto", "Diaz")
    assert [s.last_names for s in db.list_students()] == ["Diaz", "Lopez", "Ruiz"]


def test_names_sort_ignoring_accents_and_capitals(db):
    for sid, last in enumerate(["Ruiz", "Ávila", "lopez", "Diaz", "Lozano", "Ñúñez", "Nava"], start=1):
        db.add_student(sid, "X", last)
    assert [s.last_names for s in db.list_students()] == \
        ["Ávila", "Diaz", "lopez", "Lozano", "Nava", "Ñúñez", "Ruiz"]


def test_search_students(db):
    db.add_student(2024001, "Ana Maria", "Lopez")
    db.add_student(2024002, "Beto", "Diaz")
    db.add_student(2023999, "Carla", "Lopez_Ruiz")
    assert [s.id for s in db.list_students("lopez")] == [2024001, 2023999]  # case-insensitive
    assert [s.id for s in db.list_students("2024")] == [2024002, 2024001]   # by account number
    assert [s.id for s in db.list_students("maria lopez")] == [2024001]     # full name
    assert [s.id for s in db.list_students("_")] == [2023999]               # "_" is not a wildcard
    assert db.list_students("%") == []                                      # "%" is not a wildcard
    assert len(db.list_students("   ")) == 3                                # blank = everyone


# ---------- face embeddings ----------

def test_embeddings_round_trip(db):
    db.add_student(1, "Ana", "Lopez")
    db.add_student(2, "Beto", "Diaz")
    a = [random_embedding(i) for i in range(3)]
    b = [random_embedding(10).reshape(1, 128)]  # SFace's real output shape
    assert db.add_embeddings(1, a) == 3
    assert db.add_embeddings(2, b) == 1

    ids, matrix = db.all_embeddings()
    assert ids.tolist() == [1, 1, 1, 2]
    assert matrix.shape == (4, 128) and matrix.dtype == np.float32
    assert np.array_equal(matrix[0], a[0])   # stored exactly, no precision lost
    assert db.embedding_count(1) == 3


def test_no_embeddings_gives_an_empty_matrix(db):
    ids, matrix = db.all_embeddings()
    assert ids.shape == (0,) and matrix.shape == (0, 128)


@pytest.mark.parametrize("bad", [np.zeros(127), np.zeros(129), np.full(128, np.nan), np.full(128, np.inf)])
def test_invalid_embeddings_are_rejected(db, bad):
    db.add_student(1, "Ana", "Lopez")
    with pytest.raises(ValueError):
        db.add_embeddings(1, [random_embedding(0), bad])
    assert db.embedding_count(1) == 0  # nothing saved if any one is invalid


def test_embeddings_need_an_existing_student(db):
    with pytest.raises(KeyError):
        db.add_embeddings(999, [random_embedding(0)])


def test_replace_embeddings(db):
    db.add_student(1, "Ana", "Lopez")
    db.add_embeddings(1, [random_embedding(i) for i in range(5)])
    db.replace_embeddings(1, [random_embedding(99)])
    ids, matrix = db.all_embeddings()
    assert ids.tolist() == [1]
    assert np.array_equal(matrix[0], random_embedding(99))


def test_failed_replace_keeps_the_old_embeddings(db):
    db.add_student(1, "Ana", "Lopez")
    db.add_embeddings(1, [random_embedding(0)])
    with pytest.raises(ValueError):
        db.replace_embeddings(1, [np.zeros(5)])
    assert db.embedding_count(1) == 1


def test_deleting_a_student_deletes_their_face_data_and_attendance(db):
    db.add_student(1, "Ana", "Lopez")
    db.add_student(2, "Beto", "Diaz")
    db.add_embeddings(1, [random_embedding(0), random_embedding(1)])
    db.add_embeddings(2, [random_embedding(2)])
    session = db.start_session()
    db.mark_present(session, 1, 0.8)

    assert db.delete_student(1)
    assert not db.delete_student(1)  # already gone
    assert db.all_embeddings()[0].tolist() == [2]
    remaining = db.conn.execute("SELECT COUNT(*) FROM attendance").fetchone()[0]
    assert remaining == 0


# ---------- sessions and attendance ----------

def test_session_lifecycle(db, clock):
    session_id = db.start_session()
    assert db.get_session(session_id).is_open
    clock.advance(minutes=50)
    db.end_session(session_id)
    session = db.get_session(session_id)
    assert not session.is_open
    assert session.ended_at == "2026-01-12 08:50:00"
    with pytest.raises(ValueError):
        db.end_session(session_id)  # can't end twice


def test_mark_present_once_per_session(db, clock):
    db.add_student(1, "Ana", "Lopez")
    clock.advance(minutes=1)
    session = db.start_session()
    clock.advance(minutes=2)
    assert db.mark_present(session, 1, 0.55) is True    # first time
    clock.advance(minutes=3)
    assert db.mark_present(session, 1, 0.70) is False   # already present
    assert db.mark_present(session, 1, 0.40) is False

    row = db.session_report(session)[0]
    assert row.seen_at == "2026-01-12 08:03:00"         # first time seen is kept
    assert row.similarity == pytest.approx(0.70)        # best score is kept


def test_cannot_mark_present_in_a_closed_or_missing_session(db):
    db.add_student(1, "Ana", "Lopez")
    session = db.start_session()
    db.end_session(session)
    with pytest.raises(ValueError):
        db.mark_present(session, 1, 0.9)
    with pytest.raises(ValueError):
        db.mark_present(12345, 1, 0.9)


def test_cannot_mark_an_unknown_student(db):
    session = db.start_session()
    with pytest.raises(sqlite3.IntegrityError):
        db.mark_present(session, 999, 0.9)


def test_report_lists_present_first_then_absent(db, clock):
    db.add_student(1, "Ana", "Lopez")
    db.add_student(2, "Beto", "Diaz")
    db.add_student(3, "Carla", "Ruiz")
    clock.advance(minutes=1)
    session = db.start_session()
    db.mark_present(session, 3, 0.9)
    db.mark_present(session, 1, 0.8)

    report = db.session_report(session)
    assert [(r.student.id, r.present) for r in report] == [(1, True), (3, True), (2, False)]
    absent = report[2]
    assert absent.seen_at is None and absent.similarity is None
    assert db.get_session(session).present == 2
    assert db.get_session(session).enrolled == 3


def test_students_registered_later_are_not_absent_in_old_sessions(db, clock):
    db.add_student(1, "Ana", "Lopez")
    clock.advance(minutes=1)
    old = db.start_session()
    db.end_session(old)
    clock.advance(days=1)
    db.add_student(2, "Beto", "Diaz")  # joined the class after the first session

    assert [r.student.id for r in db.session_report(old)] == [1]
    assert db.get_session(old).enrolled == 1


def test_student_registered_during_a_session_counts_if_recognized(db, clock):
    session = db.start_session()
    clock.advance(minutes=5)
    db.add_student(1, "Ana", "Lopez")
    db.mark_present(session, 1, 0.9)
    assert [r.present for r in db.session_report(session)] == [True]
    assert db.get_session(session).enrolled == 1


def test_list_sessions_newest_first(db, clock):
    first = db.start_session()
    clock.advance(days=1)
    second = db.start_session()
    assert [s.id for s in db.list_sessions()] == [second, first]


def test_delete_session(db):
    db.add_student(1, "Ana", "Lopez")
    session = db.start_session()
    db.mark_present(session, 1, 0.9)
    assert db.delete_session(session)
    assert db.get_session(session) is None
    assert db.conn.execute("SELECT COUNT(*) FROM attendance").fetchone()[0] == 0
    with pytest.raises(KeyError):
        db.session_report(session)


def test_student_stats(db, clock):
    db.add_student(1, "Ana", "Lopez")
    for day in range(4):                 # 4 sessions; Ana comes to 3
        clock.advance(days=1)
        session = db.start_session()
        if day != 2:
            db.mark_present(session, 1, 0.9)
        db.end_session(session)
    clock.advance(days=1)
    db.add_student(2, "Beto", "Diaz")    # registered after all of them

    stats = {s.student.id: s for s in db.student_stats()}
    assert (stats[1].attended, stats[1].total) == (3, 4)
    assert stats[1].percentage == pytest.approx(75.0)
    assert (stats[2].attended, stats[2].total) == (0, 0)
    assert stats[2].percentage is None   # no sessions yet: no data, not 0 %


# ---------- the file itself ----------

def test_data_survives_reopening(tmp_path, clock):
    path = tmp_path / "school.db"
    with Database(path, clock=clock) as db:
        db.add_student(1, "Ana", "Lopez")
        db.add_embeddings(1, [random_embedding(0)])
    with Database(path, clock=clock) as db:
        assert db.get_student(1).full_name == "Ana Lopez"
        assert db.embedding_count(1) == 1


def test_database_from_a_newer_version_is_refused(tmp_path):
    path = tmp_path / "future.db"
    conn = sqlite3.connect(path)
    conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION + 1}")
    conn.close()
    with pytest.raises(RuntimeError):
        Database(path)
