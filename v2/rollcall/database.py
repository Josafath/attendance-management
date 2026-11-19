"""Students, face embeddings, sessions and attendance, stored in SQLite.

Tables:

    students         one row per student; the account number is the primary key
    face_embeddings  several face "fingerprints" (128 numbers) per student, not photos
    sessions         every time attendance is taken
    attendance       who was present in each session

Absent students are not stored: they are the students enrolled when the session
started who have no attendance row for it. Deleting a student also deletes
their embeddings and attendance (ON DELETE CASCADE), so no face data is left behind.

No UI code here, so everything can be tested without a window or a camera.
"""
from __future__ import annotations

import sqlite3
import unicodedata
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional, Sequence

import numpy as np

# SFace turns a face into 128 numbers. Every embedding must have this size.
EMBEDDING_SIZE = 128

# Timestamps are stored as text in this format: it's readable and sorts correctly.
TIME_FORMAT = "%Y-%m-%d %H:%M:%S"

# Bumped when the tables change, so a future version can upgrade old databases.
SCHEMA_VERSION = 1

SCHEMA = """
CREATE TABLE IF NOT EXISTS students (
    id          INTEGER PRIMARY KEY,          -- account number, chosen by the school
    first_names TEXT NOT NULL,
    last_names  TEXT NOT NULL,
    created_at  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS face_embeddings (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    student_id  INTEGER NOT NULL REFERENCES students(id) ON DELETE CASCADE,
    vector      BLOB NOT NULL,                -- 128 float32 numbers = 512 bytes
    created_at  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS sessions (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    started_at  TEXT NOT NULL,
    ended_at    TEXT                          -- NULL while attendance is being taken
);

CREATE TABLE IF NOT EXISTS attendance (
    session_id  INTEGER NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    student_id  INTEGER NOT NULL REFERENCES students(id) ON DELETE CASCADE,
    seen_at     TEXT NOT NULL,                -- first time the student was recognized
    similarity  REAL NOT NULL,                -- best match score during the session
    PRIMARY KEY (session_id, student_id)      -- a student is present once per session
);

CREATE INDEX IF NOT EXISTS idx_embeddings_student ON face_embeddings(student_id);
CREATE INDEX IF NOT EXISTS idx_attendance_student ON attendance(student_id);
"""


class DuplicateStudentError(ValueError):
    """There is already a student with that account number."""


@dataclass(frozen=True)
class Student:
    id: int
    first_names: str
    last_names: str
    created_at: str

    @property
    def full_name(self) -> str:
        """First names first, as shown on the camera: "Ana Lopez"."""
        return f"{self.first_names} {self.last_names}"


@dataclass(frozen=True)
class Session:
    id: int
    started_at: str
    ended_at: Optional[str]
    present: int   # students recognized in this session
    enrolled: int  # students who were registered when it started (or showed up)

    @property
    def is_open(self) -> bool:
        return self.ended_at is None


@dataclass(frozen=True)
class ReportRow:
    """One line of a session's attendance list."""
    student: Student
    present: bool
    seen_at: Optional[str]        # None when absent
    similarity: Optional[float]   # None when absent


@dataclass(frozen=True)
class StudentStats:
    """How often a student came, counting only sessions since they registered."""
    student: Student
    attended: int
    total: int

    @property
    def percentage(self) -> Optional[float]:
        # None, not 0 %, when there were no sessions yet: "no data" is not "never came".
        return None if self.total == 0 else 100 * self.attended / self.total


class Database:
    """All reads and writes of the app go through this class.

    `clock` returns the current time. The app uses the real one; the tests pass
    a fake clock so they can control the dates.
    """

    def __init__(self, path: Path | str, clock: Callable[[], datetime] = datetime.now) -> None:
        self.clock = clock
        self.conn = sqlite3.connect(str(path))
        # Rows can be read by column name: row["first_names"].
        self.conn.row_factory = sqlite3.Row
        # SQLite ignores REFERENCES unless this is turned on for every connection.
        self.conn.execute("PRAGMA foreign_keys = ON")
        # "COLLATE NAME" in the queries sorts names alphabetically the way people
        # expect, ignoring accents and capitals (see _compare_names).
        self.conn.create_collation("NAME", _compare_names)
        self._create_schema()

    # ---------- setup ----------

    def _create_schema(self) -> None:
        version = self.conn.execute("PRAGMA user_version").fetchone()[0]
        if version > SCHEMA_VERSION:
            # A newer RollCall created this file; writing to it could damage it.
            raise RuntimeError(f"The database is from a newer version of RollCall (schema {version}).")
        with self.conn:  # a transaction: everything or nothing
            self.conn.executescript(SCHEMA)
            self.conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")

    def close(self) -> None:
        self.conn.close()

    # "with Database(path) as db:" closes the connection at the end.
    def __enter__(self) -> "Database":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def _now(self) -> str:
        return self.clock().strftime(TIME_FORMAT)

    # ---------- students ----------

    def add_student(self, student_id: int, first_names: str, last_names: str,
                    embeddings: Sequence[np.ndarray] = ()) -> Student:
        """Adds a student and, optionally, their face embeddings.

        Both are saved in one transaction: a student is never left half-registered.
        """
        student_id, first_names, last_names = clean_student(student_id, first_names, last_names)
        blobs = [_to_blob(v) for v in embeddings]  # validated before saving anything
        now = self._now()
        try:
            with self.conn:
                self.conn.execute(
                    "INSERT INTO students (id, first_names, last_names, created_at) VALUES (?, ?, ?, ?)",
                    (student_id, first_names, last_names, now),
                )
                self.conn.executemany(
                    "INSERT INTO face_embeddings (student_id, vector, created_at) VALUES (?, ?, ?)",
                    [(student_id, blob, now) for blob in blobs],
                )
        except sqlite3.IntegrityError:
            # The primary key already exists.
            raise DuplicateStudentError(f"There is already a student with account number {student_id}.") from None
        return self.get_student(student_id)

    def update_student(self, student_id: int, first_names: str, last_names: str) -> Student:
        """Changes the names. The account number can't change: it's the student's identity."""
        student_id, first_names, last_names = clean_student(student_id, first_names, last_names)
        with self.conn:
            cursor = self.conn.execute(
                "UPDATE students SET first_names = ?, last_names = ? WHERE id = ?",
                (first_names, last_names, student_id),
            )
        if cursor.rowcount == 0:
            raise KeyError(f"No student with account number {student_id}.")
        return self.get_student(student_id)

    def delete_student(self, student_id: int) -> bool:
        """Deletes the student with their face embeddings and attendance. False if not found."""
        with self.conn:
            cursor = self.conn.execute("DELETE FROM students WHERE id = ?", (student_id,))
        return cursor.rowcount > 0

    def get_student(self, student_id: int) -> Optional[Student]:
        row = self.conn.execute("SELECT * FROM students WHERE id = ?", (student_id,)).fetchone()
        return _student(row) if row else None

    def list_students(self, search: str = "") -> list[Student]:
        """Students sorted by last names. `search` matches the account number or any name."""
        search = search.strip()
        if not search:
            rows = self.conn.execute("SELECT * FROM students ORDER BY last_names COLLATE NAME, first_names COLLATE NAME, id")
        else:
            # % and _ are wildcards in LIKE, so they are escaped to be searched literally.
            pattern = "%" + search.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_") + "%"
            rows = self.conn.execute(
                "SELECT * FROM students "
                "WHERE CAST(id AS TEXT) LIKE ? ESCAPE '\\' "
                "   OR first_names LIKE ? ESCAPE '\\' "
                "   OR last_names LIKE ? ESCAPE '\\' "
                "   OR first_names || ' ' || last_names LIKE ? ESCAPE '\\' "
                "ORDER BY last_names COLLATE NAME, first_names COLLATE NAME, id",
                (pattern, pattern, pattern, pattern),
            )
        return [_student(row) for row in rows]

    # ---------- face embeddings ----------

    def add_embeddings(self, student_id: int, vectors: list[np.ndarray]) -> int:
        """Saves the face embeddings taken at registration. Returns how many were saved."""
        if self.get_student(student_id) is None:
            raise KeyError(f"No student with account number {student_id}.")
        blobs = [_to_blob(v) for v in vectors]  # validates all of them before saving any
        now = self._now()
        with self.conn:
            self.conn.executemany(
                "INSERT INTO face_embeddings (student_id, vector, created_at) VALUES (?, ?, ?)",
                [(student_id, blob, now) for blob in blobs],
            )
        return len(blobs)

    def replace_embeddings(self, student_id: int, vectors: list[np.ndarray]) -> int:
        """Deletes a student's embeddings and saves new ones, e.g. to retake their face."""
        blobs = [_to_blob(v) for v in vectors]
        if self.get_student(student_id) is None:
            raise KeyError(f"No student with account number {student_id}.")
        now = self._now()
        with self.conn:  # one transaction: the student is never left without a face
            self.conn.execute("DELETE FROM face_embeddings WHERE student_id = ?", (student_id,))
            self.conn.executemany(
                "INSERT INTO face_embeddings (student_id, vector, created_at) VALUES (?, ?, ?)",
                [(student_id, blob, now) for blob in blobs],
            )
        return len(blobs)

    def embedding_count(self, student_id: int) -> int:
        return self.conn.execute(
            "SELECT COUNT(*) FROM face_embeddings WHERE student_id = ?", (student_id,)
        ).fetchone()[0]

    def all_embeddings(self) -> tuple[np.ndarray, np.ndarray]:
        """Every embedding, ready to compare against a face from the camera.

        Returns (ids, matrix): ids[i] is the student of the embedding in matrix[i].
        One matrix lets NumPy compare a face with all of them in a single operation.
        """
        rows = self.conn.execute("SELECT student_id, vector FROM face_embeddings ORDER BY id").fetchall()
        ids = np.array([row["student_id"] for row in rows], dtype=np.int64)
        matrix = np.array([_from_blob(row["vector"]) for row in rows], dtype=np.float32)
        return ids, matrix.reshape(len(rows), EMBEDDING_SIZE)

    # ---------- sessions and attendance ----------

    def start_session(self) -> int:
        """Starts taking attendance. Returns the new session's id."""
        with self.conn:
            cursor = self.conn.execute("INSERT INTO sessions (started_at) VALUES (?)", (self._now(),))
        return cursor.lastrowid

    def end_session(self, session_id: int) -> None:
        with self.conn:
            cursor = self.conn.execute(
                "UPDATE sessions SET ended_at = ? WHERE id = ? AND ended_at IS NULL",
                (self._now(), session_id),
            )
        if cursor.rowcount == 0:
            raise ValueError(f"Session {session_id} doesn't exist or has already ended.")

    def mark_present(self, session_id: int, student_id: int, similarity: float) -> bool:
        """Marks a student as present. True the first time in this session, False after that.

        Later sightings keep the first time seen but update the best similarity.
        """
        session = self.conn.execute("SELECT ended_at FROM sessions WHERE id = ?", (session_id,)).fetchone()
        if session is None or session["ended_at"] is not None:
            raise ValueError(f"Session {session_id} doesn't exist or has already ended.")
        with self.conn:
            already = self.conn.execute(
                "SELECT 1 FROM attendance WHERE session_id = ? AND student_id = ?",
                (session_id, student_id),
            ).fetchone()
            # "Upsert": insert, or if the row already exists, keep the higher similarity.
            self.conn.execute(
                "INSERT INTO attendance (session_id, student_id, seen_at, similarity) VALUES (?, ?, ?, ?) "
                "ON CONFLICT (session_id, student_id) "
                "DO UPDATE SET similarity = MAX(similarity, excluded.similarity)",
                (session_id, student_id, self._now(), float(similarity)),
            )
        return already is None

    def delete_session(self, session_id: int) -> bool:
        """Deletes a session and its attendance (e.g. one started by mistake)."""
        with self.conn:
            cursor = self.conn.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
        return cursor.rowcount > 0

    def get_session(self, session_id: int) -> Optional[Session]:
        rows = self._sessions("WHERE s.id = ?", (session_id,))
        return rows[0] if rows else None

    def open_session(self) -> Optional[Session]:
        """The session still taking attendance, if any (e.g. the app closed halfway)."""
        rows = self._sessions("WHERE s.ended_at IS NULL", ())
        return rows[0] if rows else None

    def list_sessions(self) -> list[Session]:
        """All sessions, newest first, with how many students were present and enrolled."""
        return self._sessions("", ())

    def _sessions(self, where: str, params: tuple) -> list[Session]:
        # "enrolled" counts the students registered before the session started,
        # plus anyone who registered later but was still marked present.
        rows = self.conn.execute(
            f"""
            SELECT s.id, s.started_at, s.ended_at,
                   (SELECT COUNT(*) FROM attendance a WHERE a.session_id = s.id) AS present,
                   (SELECT COUNT(*) FROM students st
                     WHERE st.created_at <= s.started_at
                        OR st.id IN (SELECT student_id FROM attendance a WHERE a.session_id = s.id)
                   ) AS enrolled
            FROM sessions s {where}
            ORDER BY s.started_at DESC, s.id DESC
            """,
            params,
        )
        return [Session(r["id"], r["started_at"], r["ended_at"], r["present"], r["enrolled"]) for r in rows]

    def session_report(self, session_id: int) -> list[ReportRow]:
        """Present and absent students of a session: present first, then by last names."""
        if self.conn.execute("SELECT 1 FROM sessions WHERE id = ?", (session_id,)).fetchone() is None:
            raise KeyError(f"No session {session_id}.")
        rows = self.conn.execute(
            """
            SELECT st.*, a.seen_at, a.similarity
            FROM students st
            JOIN sessions s ON s.id = ?
            LEFT JOIN attendance a ON a.session_id = s.id AND a.student_id = st.id
            WHERE st.created_at <= s.started_at OR a.student_id IS NOT NULL
            ORDER BY a.student_id IS NULL, st.last_names COLLATE NAME, st.first_names COLLATE NAME, st.id
            """,
            (session_id,),
        )
        return [
            ReportRow(_student(r), r["seen_at"] is not None, r["seen_at"], r["similarity"])
            for r in rows
        ]

    def student_stats(self) -> list[StudentStats]:
        """Attendance per student, counting only sessions held since they registered."""
        rows = self.conn.execute(
            """
            SELECT st.*,
                   (SELECT COUNT(*) FROM attendance a WHERE a.student_id = st.id) AS attended,
                   (SELECT COUNT(*) FROM sessions s
                     WHERE s.started_at >= st.created_at
                        OR s.id IN (SELECT session_id FROM attendance a WHERE a.student_id = st.id)
                   ) AS total
            FROM students st
            ORDER BY st.last_names COLLATE NAME, st.first_names COLLATE NAME, st.id
            """
        )
        return [StudentStats(_student(r), r["attended"], r["total"]) for r in rows]


# ---------- helpers ----------

def clean_student(student_id, first_names: str, last_names: str) -> tuple[int, str, str]:
    """Validates and tidies a student's data. Raises ValueError with a readable message."""
    # bool is a subclass of int in Python, so True would pass as 1 without this check.
    if isinstance(student_id, bool) or not isinstance(student_id, int) or student_id <= 0:
        raise ValueError("The account number must be a positive whole number.")
    # " ".join(split()) removes extra spaces: "  ana   maria " -> "ana maria".
    first_names = " ".join(str(first_names).split())
    last_names = " ".join(str(last_names).split())
    if not first_names or not last_names:
        raise ValueError("Write the first name(s) and last name(s).")
    return student_id, first_names, last_names


def _name_key(name: str) -> tuple[str, str]:
    """How a name sorts: "Ávila" next to "Avila", "lopez" next to "López".

    NFD splits "Á" into "A" + a combining accent; dropping the accents and
    casefolding gives the primary key. The original text breaks ties, so the
    order is always the same.
    """
    decomposed = unicodedata.normalize("NFD", name)
    plain = "".join(c for c in decomposed if not unicodedata.combining(c))
    return plain.casefold(), name


def _compare_names(a: str, b: str) -> int:
    # SQLite expects a negative number, zero or a positive number, like C's strcmp.
    key_a, key_b = _name_key(a), _name_key(b)
    return (key_a > key_b) - (key_a < key_b)


def _student(row: sqlite3.Row) -> Student:
    return Student(row["id"], row["first_names"], row["last_names"], row["created_at"])


def _to_blob(vector: np.ndarray) -> bytes:
    """An embedding as 512 bytes, after checking it's 128 real numbers."""
    array = np.asarray(vector, dtype=np.float32).reshape(-1)  # SFace returns shape (1, 128)
    if array.size != EMBEDDING_SIZE:
        raise ValueError(f"An embedding must have {EMBEDDING_SIZE} numbers, not {array.size}.")
    if not np.all(np.isfinite(array)):
        raise ValueError("An embedding can't contain NaN or infinite values.")
    return array.tobytes()


def _from_blob(blob: bytes) -> np.ndarray:
    return np.frombuffer(blob, dtype=np.float32)
