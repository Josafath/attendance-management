"""Taking attendance: the use case that connects recognition with the database.

    camera frame -> AttendanceTracker (who is in the frame?) -> Database (mark present)

The UI only talks to `AttendanceSession`: it passes frames and reads results.
It never calls the tracker or the database directly for this, so the rules
(one open session at a time, a student is marked once, the session is closed
at the end) live in one place and are tested without a window or a camera.
"""
from __future__ import annotations

import queue
from dataclasses import dataclass

import numpy as np

from .database import Database, ReportRow, Session, Student
from .faces import FaceEngine
from .recognition import DEFAULT_THRESHOLD, AttendanceTracker, FrameResult, Matcher


@dataclass(frozen=True)
class RecognitionSettings:
    """How strict recognition is. The defaults are explained in recognition.py."""
    threshold: float = DEFAULT_THRESHOLD  # minimum similarity to accept a match
    needed: int = 5                       # sightings needed to mark a student present...
    window: int = 8                       # ...within this many recent frames
    min_face_size: int = 40               # smaller faces (in pixels) are too far away


class SessionError(RuntimeError):
    """An attendance session can't be started, resumed or used."""


class NoStudentsError(SessionError):
    """There is nobody to recognize: register students first."""


class AttendanceSession:
    """One run of taking attendance, from start() to finish().

    Use start() for a new session or resume() to continue an unfinished one;
    don't call the constructor directly.
    """

    def __init__(self, db: Database, engine: FaceEngine, session_id: int,
                 settings: RecognitionSettings, matcher: Matcher) -> None:
        self.db = db
        self.session_id = session_id
        self._students = {s.id: s for s in db.list_students()}
        present = {row.student.id for row in db.session_report(session_id) if row.present}
        self.tracker = AttendanceTracker(
            engine, matcher,
            threshold=settings.threshold, needed=settings.needed, window=settings.window,
            min_face_size=settings.min_face_size, already_present=present,
        )
        # Students confirmed by analyze() and waiting for record() to save them.
        self._confirmed: queue.SimpleQueue[tuple[int, float]] = queue.SimpleQueue()
        self._finished = False

    # ---------- creating ----------

    @classmethod
    def start(cls, db: Database, engine: FaceEngine,
              settings: RecognitionSettings = RecognitionSettings()) -> "AttendanceSession":
        """Starts a new session. Fails if another one is still open."""
        # Both checks happen before creating the session, so no empty session is left behind.
        matcher = _load_matcher(db)
        open_session = db.open_session()
        if open_session is not None:
            raise SessionError(f"Session {open_session.id} (started {open_session.started_at}) "
                               "is still open. Resume it or finish it first.")
        return cls(db, engine, db.start_session(), settings, matcher)

    @classmethod
    def resume(cls, db: Database, engine: FaceEngine, session_id: int,
               settings: RecognitionSettings = RecognitionSettings()) -> "AttendanceSession":
        """Continues an unfinished session. Students already marked stay present."""
        session = db.get_session(session_id)
        if session is None or not session.is_open:
            raise SessionError(f"Session {session_id} doesn't exist or has already finished.")
        return cls(db, engine, session_id, settings, _load_matcher(db))

    # ---------- running ----------

    @property
    def threshold(self) -> float:
        return self.tracker.threshold

    @threshold.setter
    def threshold(self, value: float) -> None:
        # Cosine similarity is between -1 and 1; outside 0..1 makes no sense here.
        if not 0.0 <= value <= 1.0:
            raise ValueError("The threshold must be between 0 and 1.")
        self.tracker.threshold = value

    def process(self, frame: np.ndarray) -> FrameResult:
        """Recognizes the faces in one frame and saves the students confirmed in it."""
        result = self.analyze(frame)
        self.record()
        return result

    # process() is split in two so the UI can run the slow part in a background
    # thread: analyze() never touches the database (SQLite connections can only
    # be used by the thread that created them), and record() is quick.
    #
    # Confirmations travel from one to the other through a thread-safe queue,
    # not inside the FrameResult: the UI only draws the newest frame and skips
    # older ones, and a confirmation in a skipped frame must not be lost.

    def analyze(self, frame: np.ndarray) -> FrameResult:
        """Recognition only: who is in this frame? Safe to call from a background thread."""
        if self._finished:
            raise SessionError("This session has already finished.")
        result = self.tracker.process(frame)
        for confirmation in result.newly_confirmed:
            self._confirmed.put(confirmation)
        return result

    def record(self) -> list[tuple[int, float]]:
        """Saves every student confirmed since the last call. Returns them.

        Call it from the thread that owns the database.
        """
        if self._finished:
            raise SessionError("This session has already finished.")
        saved = []
        while True:
            try:
                student_id, similarity = self._confirmed.get_nowait()
            except queue.Empty:
                return saved
            self.db.mark_present(self.session_id, student_id, similarity)
            saved.append((student_id, similarity))

    def student(self, student_id: int) -> Student:
        """The student behind an id in a FrameResult, to show their name."""
        return self._students[student_id]

    def report(self) -> list[ReportRow]:
        """Present and absent students so far (present first)."""
        return self.db.session_report(self.session_id)

    def finish(self) -> Session:
        """Closes the session. Calling it again does nothing."""
        if not self._finished:
            self.record()  # save anyone confirmed in the very last frames
            self.db.end_session(self.session_id)
            self._finished = True
        return self.db.get_session(self.session_id)

    # "with AttendanceSession.start(...) as session:" always closes the session,
    # even if the camera fails halfway.
    def __enter__(self) -> "AttendanceSession":
        return self

    def __exit__(self, *exc) -> None:
        self.finish()


def _load_matcher(db: Database) -> Matcher:
    """Every student's face embeddings, read once when the session starts.

    Students registered while a session runs are recognized from the next one.
    """
    matcher = Matcher.from_database(db)
    if len(matcher) == 0:
        raise NoStudentsError("Register at least one student before taking attendance.")
    return matcher
