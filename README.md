# Attendance Management

A desktop app that takes class attendance with **face recognition**. Register each student once with the webcam, then point the camera at the class: the app recognizes who is there and saves the attendance list as an Excel file.

Built with Python, Tkinter, OpenCV and SQLite.

https://user-images.githubusercontent.com/29525443/138616417-8d70ac92-34df-43eb-8a17-ccb4489c6d92.mp4

## How it works

1. **Register Student.** Type the student's account number and name. The webcam takes 20 pictures of their face, and the model is retrained right away.
2. **Take Attendance.** The webcam opens and marks each face: green with the student's id if it recognizes them, red "Unknown" if not. Press **ESC** to finish.
3. The list of students present is shown on screen and saved as `Attendance_<date>_<time>.xlsx` in your Downloads folder (or your home folder if there is no Downloads folder).

Under the hood:
- **Face detection:** Haar cascade (`haarcascade_frontalface_default.xml`) finds the faces in each frame.
- **Face recognition:** LBPH (Local Binary Patterns Histograms) from `opencv-contrib`. It returns a distance: the smaller, the more similar. A face counts as recognized below `MAX_DISTANCE` (70) in `Recognition_files/recognitionFaces.py`.
- **Storage:** students in SQLite (`Students.db`), face pictures in `Recognition_files/dataset/`, trained model in `Recognition_files/trainer/trainer.yml`.

## Install and run

You need **Python 3.9 or newer** with Tkinter, and a webcam.

| System | Get Python with Tkinter |
|---|---|
| Windows | Install from [python.org](https://www.python.org/downloads/). Keep **"tcl/tk and IDLE"** ticked and tick **"Add python.exe to PATH"**. |
| macOS | Install from [python.org](https://www.python.org/downloads/), which includes Tk, or with Homebrew: `brew install python python-tk`. |
| Linux | `sudo apt install python3-tk python3-venv` (Debian/Ubuntu) or your distro's equivalent. |

Windows (PowerShell):

```powershell
cd v1
py -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python main.py
```

macOS / Linux:

```bash
cd v1
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python main.py
```

> [!IMPORTANT]
> Install `opencv-contrib-python`, not `opencv-python`. The face recognizer (`cv2.face`) is only in the contrib package. If you already have `opencv-python`, uninstall it first: both packages install `cv2` and they conflict.

On macOS, the first time you run it, allow the Terminal to use the camera (System Settings → Privacy & Security → Camera).

## Privacy

The app saves **pictures of faces** and the students' names on your computer. These files are in `.gitignore` and must never be uploaded: `Recognition_files/dataset/`, `Recognition_files/trainer/`, `Students.db` and the attendance Excel files.

## About v1

`v1/` is the original 2021 version, with the bugs that stopped it from working fixed:

- **With two or more students, the attendance list was empty.** The ids were joined into one text (`IN ('1,2,3')`), so the query only matched when exactly one student was recognized. Now each id is a query parameter.
- **Strangers were marked present.** Any face with an LBPH distance under 100 was accepted. The limit is now 70.
- **It only ran inside the IDE.** The imports mixed `App` and `src.Recognition_files`, and the file paths depended on the working folder. Now `python main.py` works from any folder.
- **Saving the Excel file crashed** on systems without a `~/Downloads` folder (on a Spanish Linux it's `~/Descargas`).
- **New students weren't recognized** unless you pressed "<- Back", the only place where the model was trained. Now it trains right after each registration.
- **Training lost pictures.** It ran the face detector again on pictures that were already cropped faces, and it often found nothing.
- **Taking attendance before registering anyone** crashed (no model and no table yet).
- **Registering** crashed with an empty or non-numeric account number or a duplicate one, and saved the student even if the camera failed.
- If two people were in front of the camera during registration, **both faces were saved with the same id**.
- The results table stayed on screen after going to Register and **kept adding rows** on every attendance.
- A second attendance on the same day **overwrote** the first Excel file.
- `.gitignore` didn't exclude the face pictures, the model or the database.

## License

[MIT](LICENSE)
