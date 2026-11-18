"""Fake camera frames and a fake face engine, so recognition can be tested without models.

A fake "frame" is a list of FakeFace: what the engine should find in it.
"""
from dataclasses import dataclass

import numpy as np

from rollcall.faces import Face, normalize


@dataclass(frozen=True)
class FakeFace:
    embedding: np.ndarray
    size: int = 120
    score: float = 0.95
    x: int = 0


class FakeEngine:
    """Implements the FaceEngine protocol: detect() and embed()."""

    def __init__(self) -> None:
        self.embed_calls = 0

    def detect(self, frame: list[FakeFace]) -> list[Face]:
        # raw holds the FakeFace itself, so embed() can return its embedding.
        return [Face((f.x, 0, f.size, f.size), f.score, f) for f in frame]

    def embed(self, frame, face: Face) -> np.ndarray:
        self.embed_calls += 1
        return normalize(face.raw.embedding)


def unit(seed: int) -> np.ndarray:
    """A random direction in 128 dimensions: two different seeds are almost unrelated (cosine ~0)."""
    return normalize(np.random.default_rng(seed).normal(size=128))


def near(vector: np.ndarray, similarity: float, seed: int = 0) -> np.ndarray:
    """A vector with exactly this cosine similarity to `vector` (same person, another photo)."""
    vector = normalize(vector)
    other = np.random.default_rng(1000 + seed).normal(size=128)
    other -= (other @ vector) * vector          # remove the part parallel to vector
    other = normalize(other)                     # now other is perpendicular to vector
    return normalize(similarity * vector + np.sqrt(1 - similarity ** 2) * other)
