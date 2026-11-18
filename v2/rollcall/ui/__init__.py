"""The Tkinter interface. Everything else in rollcall works without it.

    app.py              the main window: setup, then three tabs
    setup.py            first run: downloads the face models
    attendance_tab.py   take attendance with the camera
    students_tab.py     list, search, edit and delete students
    register_dialog.py  register a new student or retake a face
    history_tab.py      past sessions, reports and Excel export
    video.py            shows camera frames (without Pillow) and polls the worker
    theme.py            colors, fonts and ttk styles
"""
