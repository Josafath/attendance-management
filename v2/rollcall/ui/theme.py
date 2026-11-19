"""Colors, fonts and ttk styles, in one place so every screen looks the same."""
from __future__ import annotations

import tkinter as tk
from tkinter import ttk

BG = "#f6f7fb"        # window background
PANEL = "#ffffff"     # cards and lists
TEXT = "#1f2933"
MUTED = "#6b7280"
ACCENT = "#2563eb"    # main buttons
PRESENT = "#16a34a"   # green: confirmed present
PENDING = "#f59e0b"   # amber: recognized, still confirming
UNKNOWN = "#dc2626"   # red: not recognized
VIDEO_BG = "#111827"  # behind the camera image

FONT = "TkDefaultFont"


def apply(root: tk.Tk) -> None:
    """Sets up the ttk styles used by the app."""
    style = ttk.Style(root)
    # "clam" looks the same on Windows, macOS and Linux and accepts custom colors.
    if "clam" in style.theme_names():
        style.theme_use("clam")
    root.configure(bg=BG)
    style.configure(".", background=BG, foreground=TEXT)
    style.configure("TFrame", background=BG)
    style.configure("Card.TFrame", background=PANEL)
    style.configure("TLabel", background=BG, foreground=TEXT)
    style.configure("Card.TLabel", background=PANEL)
    style.configure("Muted.TLabel", foreground=MUTED)
    style.configure("Title.TLabel", font=(FONT, 16, "bold"))
    style.configure("Big.TLabel", font=(FONT, 22, "bold"))
    style.configure("Hint.TLabel", font=(FONT, 12))
    style.configure("Accent.TButton", background=ACCENT, foreground="white", padding=(14, 6))
    style.map("Accent.TButton", background=[("disabled", "#93b4f5"), ("active", "#1d4ed8")])
    style.configure("TButton", padding=(10, 5))
    style.configure("Treeview", rowheight=24, background=PANEL, fieldbackground=PANEL)
    style.configure("Treeview.Heading", font=(FONT, 10, "bold"))
    style.configure("TNotebook.Tab", padding=(16, 6))
