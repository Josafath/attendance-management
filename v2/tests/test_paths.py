from pathlib import Path

from rollcall import paths


def test_env_var_overrides_the_data_folder(isolated_data_dir):
    assert paths.data_dir() == isolated_data_dir
    assert isolated_data_dir.is_dir()  # created on first use


def test_files_live_inside_the_data_folder(isolated_data_dir):
    assert paths.database_file().parent == isolated_data_dir
    assert paths.models_dir() == isolated_data_dir / "models"
    assert paths.models_dir().is_dir()


def test_default_folder_per_operating_system(monkeypatch, tmp_path):
    monkeypatch.delenv("ROLLCALL_DATA_DIR")
    monkeypatch.setattr(Path, "home", lambda: tmp_path)

    monkeypatch.setattr(paths.sys, "platform", "win32")
    monkeypatch.setenv("APPDATA", str(tmp_path / "Roaming"))
    assert paths.data_dir() == tmp_path / "Roaming" / "RollCall"

    monkeypatch.setattr(paths.sys, "platform", "darwin")
    assert paths.data_dir() == tmp_path / "Library" / "Application Support" / "RollCall"

    monkeypatch.setattr(paths.sys, "platform", "linux")
    monkeypatch.delenv("XDG_DATA_HOME", raising=False)
    assert paths.data_dir() == tmp_path / ".local" / "share" / "rollcall"
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "xdg"))
    assert paths.data_dir() == tmp_path / "xdg" / "rollcall"


def test_exports_go_to_downloads_or_home(monkeypatch, tmp_path):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    assert paths.exports_dir() == tmp_path  # no Downloads folder (e.g. "Descargas")
    (tmp_path / "Downloads").mkdir()
    assert paths.exports_dir() == tmp_path / "Downloads"
