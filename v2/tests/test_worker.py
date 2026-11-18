import threading
import time

import numpy as np
import pytest

from rollcall.worker import MAX_EMPTY_READS, FrameWorker


class FakeCamera:
    """Returns numbered frames; after `frames` of them it returns None (camera unplugged)."""

    def __init__(self, frames: int = 10_000, delay: float = 0.001) -> None:
        self.count, self.frames, self.delay = 0, frames, delay
        self.released = threading.Event()

    def read(self):
        time.sleep(self.delay)
        if self.count >= self.frames:
            return None
        self.count += 1
        return np.full((4, 4, 3), self.count % 256, dtype=np.uint8)

    def release(self):
        self.released.set()


def wait_for(condition, timeout=3.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if condition():
            return True
        time.sleep(0.005)
    return False


def test_worker_analyzes_frames_in_the_background():
    camera = FakeCamera()
    main_thread = threading.get_ident()
    threads = set()

    def analyze(frame):
        threads.add(threading.get_ident())
        return int(frame[0, 0, 0])

    worker = FrameWorker(lambda: camera, analyze).start()
    assert wait_for(lambda: camera.count > 5)
    frame, result = worker.latest()
    assert result == frame[0, 0, 0]
    assert main_thread not in threads  # never ran in the caller's thread
    worker.stop()
    assert camera.released.is_set()
    assert not worker.running


def test_only_the_newest_result_is_kept():
    camera = FakeCamera()
    worker = FrameWorker(lambda: camera, lambda f: int(f[0, 0, 0])).start()
    assert wait_for(lambda: camera.count > 20)
    first = worker.latest()
    second = worker.latest()
    # Right after reading, there's either nothing new or a NEWER frame, never the same one again.
    assert second is None or second[0] is not first[0]
    worker.stop()


def test_camera_that_cannot_open_reports_the_error():
    def broken():
        raise RuntimeError("no camera")
    worker = FrameWorker(broken, lambda f: f).start()
    assert wait_for(lambda: not worker.running)
    assert str(worker.error) == "no camera"


def test_camera_that_stops_sending_images():
    camera = FakeCamera(frames=3, delay=0)
    worker = FrameWorker(lambda: camera, lambda f: None).start()
    assert wait_for(lambda: not worker.running)
    assert "stopped sending images" in str(worker.error)
    assert camera.released.is_set()  # released even after an error
    assert MAX_EMPTY_READS > 1


def test_analyze_errors_are_reported_and_release_the_camera():
    camera = FakeCamera()
    worker = FrameWorker(lambda: camera, lambda f: 1 / 0).start()
    assert wait_for(lambda: not worker.running)
    assert isinstance(worker.error, ZeroDivisionError)
    assert camera.released.is_set()
