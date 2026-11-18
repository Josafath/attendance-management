"""What every screen needs, passed around as one object instead of globals."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from ..camera import Camera
from ..database import Database
from ..faces import FaceEngine


@dataclass
class Context:
    db: Database
    engine: FaceEngine
    open_camera: Callable[[], Camera]
    # Only one screen can use the camera at a time (attendance OR registration).
    camera_in_use: bool = False
    _listeners: list[Callable[[], None]] = field(default_factory=list)

    def on_change(self, callback: Callable[[], None]) -> None:
        """Calls `callback` whenever students or sessions change (to refresh lists)."""
        self._listeners.append(callback)

    def changed(self) -> None:
        """Tells every screen that the data changed."""
        for callback in self._listeners:
            callback()
