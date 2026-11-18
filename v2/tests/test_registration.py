import pytest

from rollcall.database import DuplicateStudentError
from rollcall.recognition import EnrollStatus
from rollcall.registration import Registration, parse_account_number
from fakes import FakeEngine, FakeFace, near, unit

ANA, BETO = unit(1), unit(2)


def capture(registration, person, frames=40):
    """Feeds varied pictures of one person until the face is fully captured."""
    for i in range(frames):
        step = registration.analyze([FakeFace(near(person, 0.85, i))])
        if step.status == EnrollStatus.DONE:
            return
    raise AssertionError("never finished")


@pytest.mark.parametrize("text, expected", [("2024001", 2024001), ("  42 ", 42)])
def test_parse_account_number(text, expected):
    assert parse_account_number(text) == expected


@pytest.mark.parametrize("text", ["", "abc", "-5", "1.5", "12a", "   "])
def test_parse_account_number_rejects(text):
    with pytest.raises(ValueError):
        parse_account_number(text)


def test_check_validates_before_the_camera_starts(db):
    assert Registration.check(db, " 101 ", " Ana ", "Lopez") == (101, "Ana", "Lopez")
    with pytest.raises(ValueError):
        Registration.check(db, "101", "", "Lopez")
    db.add_student(101, "Ana", "Lopez")
    with pytest.raises(DuplicateStudentError):
        Registration.check(db, "101", "Other", "Person")


def test_register_a_new_student(db):
    registration = Registration(db, FakeEngine(), samples=4)
    assert not registration.done
    capture(registration, ANA)
    student = registration.save(101, "Ana", "Lopez")
    assert student.full_name == "Ana Lopez"
    assert db.embedding_count(101) == 4


def test_cannot_save_before_the_face_is_captured(db):
    with pytest.raises(RuntimeError):
        Registration(db, FakeEngine()).save(101, "Ana", "Lopez")
    assert db.get_student(101) is None


def test_student_and_face_are_saved_together(db):
    registration = Registration(db, FakeEngine(), samples=2)
    capture(registration, ANA)
    db.add_student(101, "Someone", "Else")  # the number was taken meanwhile
    with pytest.raises(DuplicateStudentError):
        registration.save(101, "Ana", "Lopez")
    assert db.embedding_count(101) == 0  # no orphan face data


def test_warns_when_the_face_is_already_registered(db):
    first = Registration(db, FakeEngine(), samples=3)
    capture(first, ANA)
    first.save(101, "Ana", "Lopez")

    same_person = Registration(db, FakeEngine(), samples=3)
    capture(same_person, ANA)
    student, similarity = same_person.duplicate()
    assert student.id == 101 and similarity > 0.5

    other_person = Registration(db, FakeEngine(), samples=3)
    capture(other_person, BETO)
    assert other_person.duplicate() is None


def test_retake_replaces_the_face_and_ignores_itself(db):
    first = Registration(db, FakeEngine(), samples=3)
    capture(first, ANA)
    first.save(101, "Ana", "Lopez")

    retake = Registration(db, FakeEngine(), retake=101, samples=5)
    capture(retake, ANA)
    assert retake.duplicate() is None  # it's her own face: not a duplicate
    assert retake.save().id == 101
    assert db.embedding_count(101) == 5


def test_retake_of_missing_student(db):
    with pytest.raises(KeyError):
        Registration(db, FakeEngine(), retake=999)
