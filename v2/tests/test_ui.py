"""The whole app with a fake camera and a fake face engine, driven like a user would.

Windows are hidden while the test runs. Skipped where there's no display.
"""
import os
import threading
import time
import tkinter as tk
from tkinter import filedialog, messagebox

import numpy as np
import pytest

from rollcall.faces import Face
from rollcall.ui.app import App
from rollcall.ui.register_dialog import RegisterDialog
from fakes import near, unit

if os.environ.get("ROLLCALL_SKIP_UI_TESTS"):
    # CI runs these on Linux with a virtual display; Windows and macOS runners
    # don't always give Tkinter a usable desktop.
    pytest.skip("UI tests disabled by ROLLCALL_SKIP_UI_TESTS", allow_module_level=True)
try:
    tk.Tk().destroy()
except tk.TclError:
    pytest.skip("no display available for Tkinter", allow_module_level=True)

PEOPLE = {1: unit(1), 2: unit(2), 3: unit(3)}  # 1 Ana, 2 Beto, 3 a stranger
BOTH = 9                                        # a frame with Ana and Beto together


class Scene:
    """What the fake camera is looking at. Tests change `who` while the app runs."""
    who = 0
    fail = False


class UiCamera:
    def __init__(self, scene: Scene) -> None:
        if scene.fail:
            raise RuntimeError("Couldn't open camera 0.")
        self.scene = scene

    def read(self):
        time.sleep(0.01)  # ~100 fps
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        frame[0, 0, 0] = self.scene.who  # the fake engine reads who is there from this pixel
        return frame

    def release(self):
        pass


class UiEngine:
    """Finds the people encoded in the frame; every embedding is slightly different."""

    def __init__(self) -> None:
        self.calls = 0
        self.lock = threading.Lock()

    def detect(self, frame):
        code = int(frame[0, 0, 0])
        people = [1, 2] if code == BOTH else ([code] if code else [])
        return [Face((50 + 250 * i, 100, 150, 150), 0.95, person) for i, person in enumerate(people)]

    def embed(self, frame, face):
        with self.lock:
            self.calls += 1
            seed = self.calls
        return near(PEOPLE[face.raw], 0.85, seed)


@pytest.fixture
def scene():
    return Scene()


@pytest.fixture
def make_app(db, scene, monkeypatch):
    # Dialogs are answered by the test, and windows stay hidden.
    monkeypatch.setattr(tk.Toplevel, "grab_set", lambda self: None)
    monkeypatch.setattr(messagebox, "showinfo", lambda *a, **k: None)
    created = []

    def make():
        app = App(db, open_camera=lambda: UiCamera(scene), make_engine=UiEngine, needs_setup=lambda: False)
        app.withdraw()
        created.append(app)
        return app

    yield make
    for app in created:
        try:
            if app.attendance and app.attendance.running:
                app.attendance.shutdown()
            app.destroy()
        except tk.TclError:
            pass


def pump(app, condition, timeout=5.0):
    """Runs Tkinter's event loop until `condition()` is true."""
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        app.update()
        if condition():
            return
        time.sleep(0.005)
    raise AssertionError("timed out waiting for the app")


def register(app, scene, account, first, last):
    dialog = RegisterDialog(app.students, app.ctx, on_saved=app.students._saved)
    dialog.withdraw()
    dialog.account.set(account)
    dialog.first.set(first)
    dialog.last.set(last)
    dialog.continue_to_camera()
    return dialog


def test_full_workflow(db, scene, make_app, monkeypatch, tmp_path):
    app = make_app()

    # 1. Register Ana and Beto.
    scene.who = 1
    register(app, scene, "101", "Ana María", "López")
    pump(app, lambda: db.get_student(101) is not None)
    scene.who = 2
    register(app, scene, "102", "Beto", "Díaz")
    pump(app, lambda: db.get_student(102) is not None)
    assert db.embedding_count(101) == 10
    assert len(app.students.table.get_children()) == 2
    assert not app.ctx.camera_in_use

    # 2. Registering Ana's face under another number triggers the duplicate warning.
    asked = []
    monkeypatch.setattr(messagebox, "askyesno", lambda *a, **k: asked.append(a) or False)
    scene.who = 1
    dialog = register(app, scene, "103", "Ana", "Again")
    pump(app, lambda: asked and not dialog.winfo_exists())
    assert "Ana María López" in asked[0][1]
    assert db.get_student(103) is None

    # 3. Take attendance with both in front of the camera; a stranger is ignored.
    scene.who = 3
    app.attendance.start()
    pump(app, lambda: app.attendance.video.find_all())  # the video is showing
    scene.who = BOTH
    session_id = app.attendance.session.session_id
    pump(app, lambda: db.get_session(session_id).present == 2)
    pump(app, lambda: len(app.attendance.present.get_children()) == 2)
    assert app.attendance.count.cget("text") == "2 / 2"

    # The strictness slider reaches the running session.
    app.attendance.threshold.set(0.5)
    app.attendance._threshold_changed()
    assert app.attendance.session.threshold == 0.5

    # Registration is blocked while the camera is in use.
    blocked = []
    monkeypatch.setattr(messagebox, "showinfo", lambda *a, **k: blocked.append(a))
    RegisterDialog(app.students, app.ctx, on_saved=lambda s: None, retake=db.get_student(101))
    assert "Camera busy" in blocked[0][0]

    app.attendance.finish()
    assert not db.get_session(session_id).is_open
    assert not app.ctx.camera_in_use

    # 4. Export the session to Excel.
    target = tmp_path / "out.xlsx"
    monkeypatch.setattr(filedialog, "asksaveasfilename", lambda **k: str(target))
    app.attendance.export()
    assert target.stat().st_size > 0

    # 5. History shows the session and its list.
    app.history.refresh()
    app.history.sessions.selection_set(str(session_id))
    app.update()
    assert len(app.history.detail.get_children()) == 2
    assert "2 of 2" in app.history.detail_title.cget("text")

    # 6. Students show 100 %; deleting Beto removes him everywhere.
    app.students.refresh()
    assert "100 %" in app.students.table.item("101")["values"][4]
    monkeypatch.setattr(messagebox, "askyesno", lambda *a, **k: True)
    app.students.table.selection_set("102")
    app.students.delete()
    assert db.get_student(102) is None
    assert not app.students.table.exists("102")


def test_unfinished_session_is_resumed_on_next_start(db, scene, make_app, monkeypatch):
    app = make_app()
    scene.who = 1
    register(app, scene, "101", "Ana", "Lopez")
    pump(app, lambda: db.get_student(101) is not None)

    app.attendance.start()
    session_id = app.attendance.session.session_id
    pump(app, lambda: db.get_session(session_id).present == 1)
    # Close the app and choose "No": keep the session open for later.
    monkeypatch.setattr(messagebox, "askyesnocancel", lambda *a, **k: False)
    app.close()
    assert db.open_session().id == session_id

    # Next start: answer "Yes" to continue it. Ana stays present, not marked twice.
    monkeypatch.setattr(messagebox, "askyesnocancel", lambda *a, **k: True)
    app = make_app()
    pump(app, lambda: app.attendance.running)
    assert app.attendance.session.session_id == session_id
    pump(app, lambda: len(app.attendance.present.get_children()) == 1)
    app.attendance.finish()
    assert db.get_session(session_id).present == 1


def test_camera_failure_keeps_the_session_for_later(db, scene, make_app, monkeypatch):
    app = make_app()
    scene.who = 1
    register(app, scene, "101", "Ana", "Lopez")
    pump(app, lambda: db.get_student(101) is not None)

    errors = []
    monkeypatch.setattr(messagebox, "showerror", lambda *a, **k: errors.append(a))
    scene.fail = True
    app.attendance.start()
    pump(app, lambda: errors)
    assert "Couldn't open camera" in errors[0][1]
    assert not app.ctx.camera_in_use
    assert db.open_session() is not None                 # still there, can be continued
    assert app.attendance.start_button.cget("text") == "Continue session"

    scene.fail = False
    app.attendance.start_button.invoke()
    pump(app, lambda: db.open_session().present == 1)
    app.attendance.finish()
    assert db.open_session() is None


def test_no_students_yet(db, make_app, monkeypatch):
    app = make_app()
    warnings = []
    monkeypatch.setattr(messagebox, "showwarning", lambda *a, **k: warnings.append(a))
    app.attendance.start()
    assert "Register at least one student" in warnings[0][1]
    assert db.list_sessions() == []


def test_form_errors_are_shown_without_opening_the_camera(db, make_app):
    app = make_app()
    dialog = RegisterDialog(app.students, app.ctx, on_saved=lambda s: None)
    dialog.withdraw()
    dialog.account.set("12a")
    dialog.continue_to_camera()
    assert "whole number" in dialog.form_error.cget("text")
    assert not app.ctx.camera_in_use
    dialog.cancel()
