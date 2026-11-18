"""RollCall: class attendance with face recognition.

Modules:
    paths        where the app keeps its data (database, models, exports)
    database     students, face embeddings, sessions and attendance in SQLite
    models       downloads and verifies the YuNet and SFace models
    camera       the Camera protocol and the real webcam
    faces        face detection (YuNet) and embeddings (SFace)
    recognition  matching, confirmation over several frames, and registration
    attendance   use case: one attendance session, from camera frames to the database
    export       what goes in each Excel file (one session, full history)
    xlsx         a small .xlsx writer that only uses the standard library

Layers (each one only uses the ones below it):

    attendance, export        use cases
    recognition, xlsx         logic, no I/O of their own
    database, faces, camera   storage, models and hardware
"""

__version__ = "2.0.0"
