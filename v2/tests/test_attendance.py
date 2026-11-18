import pytest

from rollcall.attendance import AttendanceSession, NoStudentsError, RecognitionSettings, SessionError
from fakes import FakeEngine, FakeFace, near, unit

ANA, BETO, CARLA = unit(1), unit(2), unit(3)
FAST = RecognitionSettings(needed=2, window=3)  # confirm after 2 sightings, to keep tests short


@pytest.fixture
def school(db, clock):
    """Three registered students with their face embeddings."""
    for sid, first, last, face in [(1, "Ana", "Lopez", ANA), (2, "Beto", "Diaz", BETO), (3, "Carla", "Ruiz", CARLA)]:
        db.add_student(sid, first, last)
        db.add_embeddings(sid, [face])
    clock.advance(minutes=1)
    return db


def test_students_seen_are_saved_as_present(school):
    with AttendanceSession.start(school, FakeEngine(), FAST) as session:
        frame = [FakeFace(near(ANA, 0.8)), FakeFace(near(CARLA, 0.7), x=300)]
        assert session.process(frame).newly_confirmed == []         # 1st sighting: not yet
        confirmed = session.process(frame).newly_confirmed          # 2nd: confirmed
        assert sorted(sid for sid, _ in confirmed) == [1, 3]
        assert session.student(1).full_name == "Ana Lopez"

    report = school.session_report(session.session_id)
    assert [(r.student.id, r.present) for r in report] == [(1, True), (3, True), (2, False)]
    assert not school.get_session(session.session_id).is_open    # closed by the "with"


def test_cannot_start_without_students(db):
    with pytest.raises(NoStudentsError):
        AttendanceSession.start(db, FakeEngine())
    assert db.list_sessions() == []  # no empty session left behind


def test_students_without_face_data_dont_count(db):
    db.add_student(1, "Ana", "Lopez")  # registered, but no embeddings
    with pytest.raises(NoStudentsError):
        AttendanceSession.start(db, FakeEngine())


def test_only_one_open_session_at_a_time(school):
    first = AttendanceSession.start(school, FakeEngine(), FAST)
    with pytest.raises(SessionError, match="still open"):
        AttendanceSession.start(school, FakeEngine(), FAST)
    first.finish()
    AttendanceSession.start(school, FakeEngine(), FAST).finish()  # now it works
    assert len(school.list_sessions()) == 2


def test_resume_an_unfinished_session(school):
    # The app closed in the middle of a session (no finish()).
    crashed = AttendanceSession.start(school, FakeEngine(), FAST)
    crashed.process([FakeFace(ANA)])
    crashed.process([FakeFace(ANA)])  # Ana is saved as present

    open_session = school.open_session()
    assert open_session.id == crashed.session_id
    resumed = AttendanceSession.resume(school, FakeEngine(), open_session.id, FAST)
    result = resumed.process([FakeFace(ANA)])
    assert result.faces[0].confirmed          # still shown as present
    assert result.newly_confirmed == []       # not marked a second time
    for _ in range(2):
        resumed.process([FakeFace(BETO)])
    resumed.finish()
    present = [r.student.id for r in school.session_report(open_session.id) if r.present]
    assert sorted(present) == [1, 2]


def test_cannot_resume_a_finished_or_missing_session(school):
    session = AttendanceSession.start(school, FakeEngine(), FAST)
    session.finish()
    with pytest.raises(SessionError):
        AttendanceSession.resume(school, FakeEngine(), session.session_id)
    with pytest.raises(SessionError):
        AttendanceSession.resume(school, FakeEngine(), 999)


def test_finished_session_rejects_frames_and_finish_is_idempotent(school):
    session = AttendanceSession.start(school, FakeEngine(), FAST)
    first = session.finish()
    assert session.finish() == first  # calling it twice is harmless
    with pytest.raises(SessionError):
        session.process([FakeFace(ANA)])


def test_threshold_slider(school):
    session = AttendanceSession.start(school, FakeEngine(), RecognitionSettings(threshold=0.6, needed=1, window=1))
    frame = [FakeFace(near(ANA, 0.5))]
    assert session.process(frame).newly_confirmed == []
    session.threshold = 0.45
    assert session.threshold == 0.45
    assert [sid for sid, _ in session.process(frame).newly_confirmed] == [1]
    for bad in (-0.1, 1.5):
        with pytest.raises(ValueError):
            session.threshold = bad
    session.finish()


def test_live_report_while_running(school):
    session = AttendanceSession.start(school, FakeEngine(), FAST)
    for _ in range(2):
        session.process([FakeFace(BETO)])
    assert [r.student.id for r in session.report() if r.present] == [2]
    session.finish()


def test_confirmations_in_skipped_frames_are_not_lost(school):
    # The UI only draws the newest frame: several analyze() calls can happen
    # before one record(). Every confirmation must still be saved.
    session = AttendanceSession.start(school, FakeEngine(), FAST)
    frames = [[FakeFace(ANA)], [FakeFace(ANA)], [FakeFace(BETO)], [FakeFace(BETO)], []]
    for frame in frames:
        session.analyze(frame)          # background thread
    saved = session.record()            # the UI's next turn
    assert sorted(sid for sid, _ in saved) == [1, 2]
    assert session.record() == []       # nothing saved twice
    session.finish()


def test_finish_saves_the_last_confirmations(school):
    session = AttendanceSession.start(school, FakeEngine(), FAST)
    session.analyze([FakeFace(CARLA)])
    session.analyze([FakeFace(CARLA)])  # confirmed, but record() was never called
    session.finish()
    assert school.get_session(session.session_id).present == 1


def test_strangers_are_never_saved(school):
    with AttendanceSession.start(school, FakeEngine(), FAST) as session:
        for _ in range(30):
            session.process([FakeFace(unit(99))])
    assert school.get_session(session.session_id).present == 0
