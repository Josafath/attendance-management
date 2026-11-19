import hashlib
import io
import urllib.request

import pytest

from rollcall import models
from rollcall.models import DownloadError, Model

CONTENT = b"fake onnx model " * 1000


def fake_model(content: bytes = CONTENT, name: str = "fake.onnx") -> Model:
    return Model(name, "https://example.com/" + name, hashlib.sha256(content).hexdigest(), len(content))


@pytest.fixture
def serve(monkeypatch):
    """Replaces the internet: urlopen returns these bytes (or raises this error)."""
    def install(payload):
        def urlopen(url, timeout):
            if isinstance(payload, Exception):
                raise payload
            return io.BytesIO(payload)
        monkeypatch.setattr(urllib.request, "urlopen", urlopen)
    return install


def test_download_checks_the_fingerprint_and_reports_progress(serve):
    serve(CONTENT)
    model = fake_model()
    progress = []
    path = models.download(model, lambda m, done, total: progress.append((done, total)))
    assert path.read_bytes() == CONTENT
    assert model.is_ready()
    assert progress[-1] == (len(CONTENT), len(CONTENT))


def test_tampered_download_is_discarded(serve):
    serve(b"something else entirely")
    model = fake_model()
    with pytest.raises(DownloadError, match="fingerprint"):
        models.download(model)
    assert not model.path().exists()
    assert not model.path().with_suffix(".part").exists()  # no half file left behind


def test_network_error_is_reported_clearly(serve):
    serve(OSError("no internet"))
    model = fake_model()
    with pytest.raises(DownloadError, match="Couldn't download"):
        models.download(model)
    assert not model.path().with_suffix(".part").exists()


def test_ensure_models_only_downloads_what_is_missing(serve, monkeypatch):
    present, missing = fake_model(name="present.onnx"), fake_model(name="missing.onnx")
    present.path().write_bytes(CONTENT)
    monkeypatch.setattr(models, "MODELS", (present, missing))
    downloaded = []
    serve(CONTENT)
    models.ensure_models(lambda m, done, total: downloaded.append(m.filename))
    assert set(downloaded) == {"missing.onnx"}
    assert models.models_ready()


def test_wrong_size_file_is_not_ready():
    model = fake_model()
    model.path().write_bytes(CONTENT[:-1])  # e.g. a download that was cut short
    assert not model.is_ready()


def test_real_models_are_pinned():
    for model in models.MODELS:
        assert model.url.startswith("https://github.com/opencv/opencv_zoo/")
        assert len(model.sha256) == 64
