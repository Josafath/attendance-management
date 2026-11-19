"""Registering a student's face: the use case behind the "Add student" and
"Retake face" windows.

    1. check()     validate the typed data BEFORE turning on the camera
    2. analyze()   collect face embeddings from camera frames (background thread)
    3. duplicate() warn if this face already belongs to another student
    4. save()      store the student and their embeddings in one transaction

Like attendance.py, analyze() never touches the database, so it can run in a
background thread; the other methods must run in the thread that owns it.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from .database import Database, DuplicateStudentError, Student, clean_student
from .faces import FaceEngine
from .recognition import DEFAULT_THRESHOLD, EnrollStep, Enroller, Matcher, find_duplicate


def parse_account_number(text: str) -> int:
    """The account number typed in a text box, as an int. Raises ValueError if invalid."""
    text = text.strip()
    # isdigit() rejects "-5", "1.5", "12a" and "" with one check.
    if not text.isdigit():
        raise ValueError("The account number must be a whole number, like 2024001.")
    return int(text)


class Registration:
    """Collects one face and saves it, for a new student or to retake an existing one."""

    def __init__(self, db: Database, engine: FaceEngine, retake: Optional[int] = None,
                 samples: int = 10) -> None:
        self.db = db
        self.retake = retake  # the student whose face is being retaken, or None for a new one
        if retake is not None and db.get_student(retake) is None:
            raise KeyError(f"No student with account number {retake}.")
        self.enroller = Enroller(engine, needed=samples)

    @staticmethod
    def check(db: Database, account_number: str, first_names: str, last_names: str) -> tuple[int, str, str]:
        """Validates a new student's data. Returns it cleaned, or raises ValueError.

        Done before the camera starts, so nobody poses for nothing.
        """
        student_id = parse_account_number(account_number)
        student_id, first_names, last_names = clean_student(student_id, first_names, last_names)
        if db.get_student(student_id) is not None:
            raise DuplicateStudentError(f"There is already a student with account number {student_id}.")
        return student_id, first_names, last_names

    # ---------- camera ----------

    def analyze(self, frame: np.ndarray) -> EnrollStep:
        """One camera frame. Safe to call from a background thread."""
        return self.enroller.process(frame)

    @property
    def done(self) -> bool:
        return self.enroller.done

    # ---------- saving ----------

    def duplicate(self, threshold: float = DEFAULT_THRESHOLD) -> Optional[tuple[Student, float]]:
        """The registered student this face looks like, if any (other than the one being retaken)."""
        ids, matrix = self.db.all_embeddings()
        others = ids != (self.retake if self.retake is not None else -1)
        match = find_duplicate(Matcher(ids[others], matrix[others]), self.enroller.embeddings, threshold)
        if match is None:
            return None
        student_id, similarity = match
        return self.db.get_student(student_id), similarity

    def save(self, student_id: Optional[int] = None, first_names: str = "", last_names: str = "") -> Student:
        """Stores the face: a new student (with their data), or the retaken one."""
        if not self.done:
            raise RuntimeError("The face hasn't been fully captured yet.")
        if self.retake is not None:
            self.db.replace_embeddings(self.retake, self.enroller.embeddings)
            return self.db.get_student(self.retake)
        return self.db.add_student(student_id, first_names, last_names, self.enroller.embeddings)
