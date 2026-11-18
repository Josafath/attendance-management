"""From faces to students: matching, confirmation and registration.

    Matcher             compares one embedding with every registered student
    Confirmer           marks a student present only after several sightings
    AttendanceTracker   the full pipeline for one camera frame during attendance
    Enroller            collects varied embeddings of a new student's face

None of these know about the camera, the database or the UI. They receive
frames and embeddings, so they can be tested with fakes.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import numpy as np

from .faces import Face, FaceEngine, normalize

# OpenCV's recommended cosine-similarity threshold for SFace: above it, two
# faces are considered the same person. The UI lets the teacher adjust it.
DEFAULT_THRESHOLD = 0.363


# ---------------------------------------------------------------- matching

class Matcher:
    """Finds the registered student most similar to a face.

    Every student has several embeddings (taken from slightly different angles).
    A face's score for a student is its BEST similarity with any of them.
    """

    def __init__(self, ids: np.ndarray, matrix: np.ndarray) -> None:
        self.ids = np.asarray(ids, dtype=np.int64)
        # Normalized again just in case, so matrix @ embedding is always a cosine.
        self.matrix = np.array([normalize(row) for row in matrix], dtype=np.float32).reshape(-1, 128)
        # unique ids, and for each embedding the position of its student in that list
        self.students, self._owner = np.unique(self.ids, return_inverse=True)

    @classmethod
    def from_database(cls, db) -> "Matcher":
        return cls(*db.all_embeddings())

    def __len__(self) -> int:
        """How many students can be recognized."""
        return len(self.students)

    def scores(self, embedding: np.ndarray) -> dict[int, float]:
        """Best similarity of this face with each student: {student_id: similarity}."""
        if len(self.students) == 0:
            return {}
        # One matrix-vector product compares the face with ALL embeddings at once.
        similarities = self.matrix @ normalize(embedding)
        best = np.full(len(self.students), -1.0, dtype=np.float32)
        # For each embedding, keep the maximum in its student's slot.
        np.maximum.at(best, self._owner, similarities)
        return {int(sid): float(score) for sid, score in zip(self.students, best)}

    def best(self, embedding: np.ndarray, threshold: float = DEFAULT_THRESHOLD) -> tuple[Optional[int], float]:
        """(student_id, similarity) of the closest student, or (None, similarity) if below the threshold."""
        scores = self.scores(embedding)
        if not scores:
            return None, 0.0
        student_id = max(scores, key=scores.get)
        similarity = scores[student_id]
        return (student_id if similarity >= threshold else None), similarity


# ---------------------------------------------------------------- confirmation

class Confirmer:
    """A student is confirmed when recognized in `needed` of the last `window` frames.

    One lucky frame is not enough: that was v1's main source of false positives.
    The window tolerates a few missed frames (a blink, a head turn) without
    starting over, which a strict "N in a row" would not.
    """

    def __init__(self, needed: int = 5, window: int = 8) -> None:
        if not 1 <= needed <= window:
            raise ValueError("Need 1 <= needed <= window.")
        self.needed, self.window = needed, window
        # For each student, a list of the last `window` frames: True = recognized.
        self.history: dict[int, deque[bool]] = {}
        self.best_similarity: dict[int, float] = {}
        self.confirmed: set[int] = set()

    def update(self, seen: dict[int, float]) -> list[tuple[int, float]]:
        """Registers one frame. `seen` = {student_id: similarity} recognized in it.

        Returns the students confirmed by THIS frame, each only once.
        """
        newly = []
        for student_id in set(self.history) | set(seen):
            frames = self.history.setdefault(student_id, deque(maxlen=self.window))
            frames.append(student_id in seen)
            if student_id in seen:
                previous = self.best_similarity.get(student_id, -1.0)
                self.best_similarity[student_id] = max(previous, seen[student_id])
            if student_id not in self.confirmed and sum(frames) >= self.needed:
                self.confirmed.add(student_id)
                newly.append((student_id, self.best_similarity[student_id]))
        return newly

    def progress(self, student_id: int) -> float:
        """From 0 to 1: how close a student is to being confirmed (for the UI)."""
        if student_id in self.confirmed:
            return 1.0
        return min(1.0, sum(self.history.get(student_id, ())) / self.needed)


# ---------------------------------------------------------------- attendance

@dataclass(frozen=True)
class LabeledFace:
    """A face in the current frame and what the tracker decided about it."""
    face: Face
    student_id: Optional[int]  # None = unknown (or too far away)
    similarity: float          # best similarity found, even if below the threshold
    confirmed: bool            # already marked present
    progress: float            # 0..1 towards confirmation


@dataclass(frozen=True)
class FrameResult:
    faces: list[LabeledFace]
    newly_confirmed: list[tuple[int, float]]  # (student_id, best similarity): mark them present


class AttendanceTracker:
    """Runs detection, matching and confirmation on each camera frame."""

    def __init__(self, engine: FaceEngine, matcher: Matcher, threshold: float = DEFAULT_THRESHOLD,
                 needed: int = 5, window: int = 8, min_face_size: int = 40,
                 already_present: set[int] = frozenset()) -> None:
        self.engine, self.matcher = engine, matcher
        self.threshold = threshold      # can be changed while running (the UI slider)
        self.min_face_size = min_face_size
        self.confirmer = Confirmer(needed, window)
        # Students already marked in this session (e.g. after reopening it) aren't confirmed twice.
        self.confirmer.confirmed |= set(already_present)

    def process(self, frame: np.ndarray) -> FrameResult:
        faces = self.engine.detect(frame)

        # 1. Score every face that is big enough against every student.
        candidates = []  # (similarity, face index, student id)
        best_per_face: dict[int, float] = {}
        for index, face in enumerate(faces):
            if face.size < self.min_face_size:
                continue  # too far away: too few pixels for a reliable match
            scores = self.matcher.scores(self.engine.embed(frame, face))
            best_per_face[index] = max(scores.values(), default=0.0)
            candidates += [(sim, index, sid) for sid, sim in scores.items() if sim >= self.threshold]

        # 2. Assign students to faces, best matches first. A student can't be two
        #    faces at once, and a face can't be two students: when two faces look
        #    like the same student, only the more similar one gets the name.
        assigned: dict[int, tuple[int, float]] = {}  # face index -> (student, similarity)
        taken: set[int] = set()
        for similarity, index, student_id in sorted(candidates, reverse=True):
            if index not in assigned and student_id not in taken:
                assigned[index] = (student_id, similarity)
                taken.add(student_id)

        # 3. Update the confirmation counters with who was seen in this frame.
        newly = self.confirmer.update({sid: sim for sid, sim in assigned.values()})

        labeled = []
        for index, face in enumerate(faces):
            if index in assigned:
                student_id, similarity = assigned[index]
                labeled.append(LabeledFace(face, student_id, similarity,
                                           student_id in self.confirmer.confirmed,
                                           self.confirmer.progress(student_id)))
            else:
                labeled.append(LabeledFace(face, None, best_per_face.get(index, 0.0), False, 0.0))
        return FrameResult(labeled, newly)


# ---------------------------------------------------------------- registration

class EnrollStatus(Enum):
    """What happened with one frame while registering a student (shown as a hint in the UI)."""
    NO_FACE = "Look at the camera."
    MANY_FACES = "Only the student being registered should be in front of the camera."
    TOO_SMALL = "Come closer to the camera."
    UNCLEAR = "Face not clear enough. Check the light."
    TOO_SIMILAR = "Move your head a little: tilt it, turn it slightly."
    WAITING = "Hold on..."
    CAPTURED = "Captured."
    DONE = "Done!"


@dataclass(frozen=True)
class EnrollStep:
    status: EnrollStatus
    captured: int
    needed: int
    face: Optional[Face] = None  # to draw the box


@dataclass
class Enroller:
    """Collects `needed` embeddings of ONE face, from slightly different angles.

    Varied samples recognize the student better later (head turned, other light)
    than many copies of the same pose, so a sample is only kept if it differs
    enough from the ones already taken.
    """
    engine: FaceEngine
    needed: int = 10
    min_face_size: int = 80       # stricter than attendance: registration sets the reference
    min_score: float = 0.9
    max_similarity: float = 0.97  # a new sample must be less similar than this to every kept one
    every: int = 3                # look at one frame out of every 3 (~10 per second)
    embeddings: list[np.ndarray] = field(default_factory=list)
    _frame: int = 0

    @property
    def done(self) -> bool:
        return len(self.embeddings) >= self.needed

    def process(self, frame: np.ndarray) -> EnrollStep:
        def step(status: EnrollStatus, face: Optional[Face] = None) -> EnrollStep:
            return EnrollStep(status, len(self.embeddings), self.needed, face)

        if self.done:
            return step(EnrollStatus.DONE)
        faces = self.engine.detect(frame)
        if not faces:
            return step(EnrollStatus.NO_FACE)
        if len(faces) > 1:
            # Never guess which face is the student: v1 saved both under one id.
            return step(EnrollStatus.MANY_FACES)
        face = faces[0]
        if face.size < self.min_face_size:
            return step(EnrollStatus.TOO_SMALL, face)
        if face.score < self.min_score:
            return step(EnrollStatus.UNCLEAR, face)

        self._frame += 1
        if self._frame % self.every != 1 and self.every > 1:
            return step(EnrollStatus.WAITING, face)

        embedding = self.engine.embed(frame, face)
        if any(float(embedding @ kept) >= self.max_similarity for kept in self.embeddings):
            return step(EnrollStatus.TOO_SIMILAR, face)
        self.embeddings.append(embedding)
        return step(EnrollStatus.DONE if self.done else EnrollStatus.CAPTURED, face)


def find_duplicate(matcher: Matcher, embeddings: list[np.ndarray],
                   threshold: float = DEFAULT_THRESHOLD) -> Optional[tuple[int, float]]:
    """Checks if a new student's face already belongs to a registered student.

    Uses the average of the new embeddings (more stable than any single one).
    Returns (student_id, similarity) of the match, or None.
    """
    if not embeddings or len(matcher) == 0:
        return None
    average = normalize(np.mean(embeddings, axis=0))
    student_id, similarity = matcher.best(average, threshold)
    return None if student_id is None else (student_id, similarity)
