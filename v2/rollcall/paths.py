"""Where RollCall keeps its files.

Everything lives in the operating system's folder for app data, never in the
project folder, so face data can't be committed to git by accident:

    Windows   %APPDATA%\\RollCall
    macOS     ~/Library/Application Support/RollCall
    Linux     $XDG_DATA_HOME/rollcall  (usually ~/.local/share/rollcall)

The environment variable ROLLCALL_DATA_DIR overrides it. The tests use it to
work in a temporary folder, and you can use it to keep the data on a USB drive.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

APP_NAME = "RollCall"
ENV_VAR = "ROLLCALL_DATA_DIR"


def data_dir() -> Path:
    """The folder for the app's data. It's created if it doesn't exist."""
    override = os.environ.get(ENV_VAR)
    if override:
        # The user (or a test) chose the folder explicitly.
        folder = Path(override).expanduser()
    elif sys.platform == "win32":
        # %APPDATA% is the roaming profile folder, e.g. C:\Users\ana\AppData\Roaming.
        # The fallback covers the rare case where the variable isn't set.
        base = os.environ.get("APPDATA") or Path.home() / "AppData" / "Roaming"
        folder = Path(base) / APP_NAME
    elif sys.platform == "darwin":
        folder = Path.home() / "Library" / "Application Support" / APP_NAME
    else:
        # Linux and other Unix systems follow the XDG standard.
        base = os.environ.get("XDG_DATA_HOME") or Path.home() / ".local" / "share"
        folder = Path(base) / APP_NAME.lower()

    # parents=True creates missing parent folders; exist_ok=True doesn't fail if it exists.
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def database_file() -> Path:
    """The SQLite database with students, face embeddings and attendance."""
    return data_dir() / "rollcall.db"


def models_dir() -> Path:
    """Folder for the face detection and recognition models (downloaded once)."""
    folder = data_dir() / "models"
    folder.mkdir(exist_ok=True)
    return folder


def exports_dir() -> Path:
    """Default folder for Excel exports: Downloads if it exists, otherwise home.

    The user can pick another folder when saving; this is only the starting point.
    """
    downloads = Path.home() / "Downloads"
    return downloads if downloads.is_dir() else Path.home()
