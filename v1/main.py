import tkinter as tk
from App import Start

class Project(tk.Tk):

    def __init__(self):
        tk.Tk.__init__(self)
        self._frame = None
        self.switch_frames(Start)
        self._style()

    def switch_frames(self, frame_class):
        new_frame = frame_class(self)
        if self._frame is not None:
            self._frame.destroy()
        self._frame = new_frame
        self._frame.pack()

    def _style(self):
        self.geometry("840x550")
        self.title("Attendance :)")
        tk.Tk.config(self, cnf=None, bg="#264653")


if __name__ == '__main__':
    attendance = Project()
    attendance.mainloop()