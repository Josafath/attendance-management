import numpy as np
import pytest

from rollcall.recognition import (DEFAULT_THRESHOLD, AttendanceTracker, Confirmer, EnrollStatus,
                                  Enroller, Matcher, find_duplicate)
from fakes import FakeEngine, FakeFace, near, unit

ANA, BETO = unit(1), unit(2)


def matcher_for(**students) -> Matcher:
    """Matcher with students given as id=[embeddings]: matcher_for(**{"1": [...]})."""
    ids, rows = [], []
    for sid, vectors in students.items():
        for v in vectors:
            ids.append(int(sid))
            rows.append(v)
    return Matcher(np.array(ids), np.array(rows))


# ---------------------------------------------------------------- Matcher

def test_near_builds_the_requested_similarity():
    assert float(near(ANA, 0.6) @ ANA) == pytest.approx(0.6, abs=1e-5)


def test_score_is_the_best_of_each_students_embeddings():
    matcher = matcher_for(**{"1": [near(ANA, 0.5, 1), near(ANA, 0.9, 2)], "2": [BETO]})
    scores = matcher.scores(ANA)
    assert scores[1] == pytest.approx(0.9, abs=1e-5)  # the closer of Ana's two embeddings
    assert abs(scores[2]) < 0.4                       # Beto is unrelated
    assert len(matcher) == 2


def test_best_respects_the_threshold():
    matcher = matcher_for(**{"1": [ANA], "2": [BETO]})
    assert matcher.best(near(ANA, 0.8))[0] == 1
    student, similarity = matcher.best(near(ANA, 0.3))  # below 0.363
    assert student is None and similarity == pytest.approx(0.3, abs=1e-5)
    assert matcher.best(near(ANA, 0.3), threshold=0.2)[0] == 1


def test_matcher_without_students():
    matcher = Matcher(np.empty(0), np.empty((0, 128)))
    assert len(matcher) == 0
    assert matcher.scores(ANA) == {}
    assert matcher.best(ANA) == (None, 0.0)


def test_matcher_normalizes_unnormalized_rows():
    matcher = matcher_for(**{"1": [ANA * 7.5]})
    assert matcher.scores(ANA)[1] == pytest.approx(1.0, abs=1e-5)


def test_matcher_from_database(db):
    db.add_student(1, "Ana", "Lopez")
    db.add_embeddings(1, [ANA])
    assert Matcher.from_database(db).best(ANA)[0] == 1


# ---------------------------------------------------------------- Confirmer

def test_one_lucky_frame_is_not_enough():
    confirmer = Confirmer(needed=5, window=8)
    assert confirmer.update({1: 0.9}) == []
    for _ in range(7):
        assert confirmer.update({}) == []
    assert 1 not in confirmer.confirmed


def test_confirmed_after_needed_frames_with_some_gaps():
    confirmer = Confirmer(needed=5, window=8)
    pattern = [1, 1, 0, 1, 0, 1]  # 4 sightings with 2 missed frames (a blink)
    for seen in pattern:
        assert confirmer.update({1: 0.7} if seen else {}) == []
    assert confirmer.progress(1) == pytest.approx(0.8)
    assert confirmer.update({1: 0.8}) == [(1, pytest.approx(0.8))]  # 5th sighting
    assert confirmer.progress(1) == 1.0


def test_sightings_outside_the_window_are_forgotten():
    confirmer = Confirmer(needed=3, window=4)
    for seen in [1, 1, 0, 0, 0, 1]:  # the first two fall out of the 4-frame window
        assert confirmer.update({1: 0.9} if seen else {}) == []
    assert confirmer.progress(1) == pytest.approx(1 / 3)


def test_confirmed_only_once_with_best_similarity():
    confirmer = Confirmer(needed=2, window=3)
    confirmer.update({1: 0.5})
    assert confirmer.update({1: 0.7}) == [(1, pytest.approx(0.7))]
    assert confirmer.update({1: 0.99}) == []  # already confirmed


def test_confirmer_rejects_impossible_settings():
    with pytest.raises(ValueError):
        Confirmer(needed=5, window=3)
    with pytest.raises(ValueError):
        Confirmer(needed=0, window=3)


# ---------------------------------------------------------------- AttendanceTracker

def run(tracker, frames):
    """Processes several frames and collects every confirmation."""
    confirmed = []
    for frame in frames:
        confirmed += tracker.process(frame).newly_confirmed
    return confirmed


def test_tracker_confirms_two_students_in_the_same_frames():
    tracker = AttendanceTracker(FakeEngine(), matcher_for(**{"1": [ANA], "2": [BETO]}), needed=3, window=5)
    frame = [FakeFace(near(ANA, 0.8, 1)), FakeFace(near(BETO, 0.6, 2), x=300)]
    confirmed = run(tracker, [frame] * 3)
    assert sorted(sid for sid, _ in confirmed) == [1, 2]  # v1 lost everyone with 2+ students


def test_stranger_is_never_confirmed():
    tracker = AttendanceTracker(FakeEngine(), matcher_for(**{"1": [ANA]}), needed=2, window=3)
    stranger = [FakeFace(unit(99))]
    assert run(tracker, [stranger] * 20) == []
    labeled = tracker.process(stranger).faces[0]
    assert labeled.student_id is None and labeled.progress == 0.0


def test_two_faces_cannot_be_the_same_student():
    tracker = AttendanceTracker(FakeEngine(), matcher_for(**{"1": [ANA]}))
    result = tracker.process([FakeFace(near(ANA, 0.5, 1)), FakeFace(near(ANA, 0.9, 2), x=300)])
    names = [f.student_id for f in result.faces]
    assert names == [None, 1]  # only the more similar face gets Ana's name


def test_far_away_faces_are_ignored():
    engine = FakeEngine()
    tracker = AttendanceTracker(engine, matcher_for(**{"1": [ANA]}), min_face_size=40, needed=1, window=1)
    result = tracker.process([FakeFace(ANA, size=25)])
    assert result.faces[0].student_id is None and result.newly_confirmed == []
    assert engine.embed_calls == 0  # not even computed: saves time


def test_threshold_can_change_while_running():
    tracker = AttendanceTracker(FakeEngine(), matcher_for(**{"1": [ANA]}), needed=1, window=1, threshold=0.6)
    frame = [FakeFace(near(ANA, 0.5))]
    assert tracker.process(frame).newly_confirmed == []
    tracker.threshold = 0.4  # the teacher moves the slider
    assert [sid for sid, _ in tracker.process(frame).newly_confirmed] == [1]


def test_students_already_present_are_not_confirmed_again():
    tracker = AttendanceTracker(FakeEngine(), matcher_for(**{"1": [ANA]}), needed=1, window=1,
                                already_present={1})
    result = tracker.process([FakeFace(ANA)])
    assert result.newly_confirmed == []
    assert result.faces[0].confirmed  # still drawn as present


def test_empty_frames_and_no_students():
    tracker = AttendanceTracker(FakeEngine(), Matcher(np.empty(0), np.empty((0, 128))))
    assert tracker.process([]).faces == []
    assert tracker.process([FakeFace(ANA)]).faces[0].student_id is None


# ---------------------------------------------------------------- Enroller

def test_enroller_collects_varied_samples():
    enroller = Enroller(FakeEngine(), needed=3, every=1)
    steps = [enroller.process([FakeFace(near(ANA, 0.8, i))]) for i in range(3)]
    assert [s.status for s in steps] == [EnrollStatus.CAPTURED, EnrollStatus.CAPTURED, EnrollStatus.DONE]
    assert enroller.done and len(enroller.embeddings) == 3
    assert enroller.process([FakeFace(ANA)]).status == EnrollStatus.DONE


@pytest.mark.parametrize("frame, status", [
    ([], EnrollStatus.NO_FACE),
    ([FakeFace(ANA), FakeFace(BETO, x=300)], EnrollStatus.MANY_FACES),
    ([FakeFace(ANA, size=50)], EnrollStatus.TOO_SMALL),
    ([FakeFace(ANA, score=0.85)], EnrollStatus.UNCLEAR),
])
def test_enroller_rejects_bad_frames(frame, status):
    enroller = Enroller(FakeEngine(), every=1)
    assert enroller.process(frame).status == status
    assert enroller.embeddings == []


def test_enroller_skips_near_identical_samples():
    enroller = Enroller(FakeEngine(), every=1, max_similarity=0.97)
    enroller.process([FakeFace(ANA)])
    assert enroller.process([FakeFace(near(ANA, 0.99, 1))]).status == EnrollStatus.TOO_SIMILAR
    assert enroller.process([FakeFace(near(ANA, 0.90, 2))]).status == EnrollStatus.CAPTURED


def test_enroller_only_looks_at_some_frames():
    engine = FakeEngine()
    enroller = Enroller(engine, every=3)
    statuses = [enroller.process([FakeFace(near(ANA, 0.8, i))]).status for i in range(6)]
    assert statuses.count(EnrollStatus.CAPTURED) == 2
    assert engine.embed_calls == 2  # the skipped frames cost nothing


def test_find_duplicate():
    matcher = matcher_for(**{"1": [ANA]})
    assert find_duplicate(matcher, [near(ANA, 0.8, i) for i in range(3)])[0] == 1
    assert find_duplicate(matcher, [near(BETO, 0.8, i) for i in range(3)]) is None
    assert find_duplicate(matcher, []) is None
    assert DEFAULT_THRESHOLD == pytest.approx(0.363)
