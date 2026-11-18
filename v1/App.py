import tkinter.font as tkFont
import sqlite3
from tkinter import ttk, messagebox
import xlsxwriter
import datetime, os
import tkinter as tk

# Fix: these imports used "src.Recognition_files", which only worked inside the IDE.
# main.py runs from this folder, so the package is imported from here.
from Recognition_files.datasetFaces import DatasetFaces
from Recognition_files.trainerFaces import Trainer, TRAINER_FILE
from Recognition_files.recognitionFaces import Recognition

# Fix: the database lives next to this file, wherever the program is run from
DATABASE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Students.db")


def connect():
    """Opens the database and creates the table if it doesn't exist yet."""
    # Fix: before, the table was only created on the Register screen, so taking
    # attendance before registering anyone crashed with "no such table".
    conexion = sqlite3.connect(DATABASE_FILE)
    conexion.execute("CREATE TABLE IF NOT EXISTS Students "
                     "(id INT PRIMARY KEY,"
                     " last_name VARCHAR(50),"
                     " name VARCHAR(50))")
    return conexion


def downloads_directory():
    """The Downloads folder if it exists, otherwise the home folder."""
    # Fix: "~/Downloads" was hard-coded and doesn't exist on every system
    # (on a Spanish Linux it's "~/Descargas"), so saving the Excel file crashed.
    downloads = os.path.join(os.path.expanduser('~'), "Downloads")
    return downloads if os.path.isdir(downloads) else os.path.expanduser('~')


class Start(tk.Frame):

    def __init__(self, master):
        tk.Frame.__init__(self, master)
        tk.Frame.config(self,cnf=None, bg="#264653")

        self.conexion = connect()
        self.cursor = self.conexion.cursor()

        tk.Button(self, text="Register Student",
                  font= ('Times 18 bold'),relief='raised',
                  bg="#f1faee",activebackground="#ffe8d6",
                  command=lambda: master.switch_frames(Register),cursor="hand2").pack(pady=20)

        tk.Button(self, text="Take Attendance",
                  font=('Times 18 bold'),relief='raised',
                  bg="#f1faee",activebackground="#ffe8d6",
                  command=self.recognition,cursor="hand2").pack(pady=20)

        # Fix: the table belongs to this screen (self), not to the main window (master).
        # Before, it stayed on screen after going to Register.
        self.tree = ttk.Treeview(self, columns=('Id', 'Last Name(s)', 'Name(s)'), show='headings')
        cols = ('Id', 'Last Name(s)', 'Name(s)')
        for col in cols:
            self.tree.heading(col, text=col)

    def recognition(self):
        # Fix: without a trained model (no students yet) the program crashed
        if not os.path.isfile(TRAINER_FILE):
            messagebox.showwarning("", "Register at least one student first.")
            return

        recognition = Recognition()
        ids_students_in_class = recognition.go()

        # Fix: before, the ids were joined into ONE text: IN ('1,2,3').
        # That only matched when exactly one student was recognized; with two or
        # more, the attendance list came out empty. Now each id is a parameter: IN (?,?,?)
        if ids_students_in_class:
            placeholders = ",".join("?" for _ in ids_students_in_class)
            self.cursor.execute(f"SELECT * FROM Students WHERE id IN ({placeholders}) "
                                f"ORDER BY last_name, name", ids_students_in_class)
            #Getting the data from query
            data = self.cursor.fetchall()
        else:
            data = []

        # Fix: the time is taken when attendance is taken, not when the screen opened,
        # and it's part of the file name, so a second attendance on the same day
        # doesn't overwrite the first one.
        tiempo = datetime.datetime.now()
        # Numeric date: it doesn't depend on the system language and the files sort by date
        name_of_file = "Attendance_" + tiempo.strftime("%Y-%m-%d_%H-%M-%S")

        #Setting path for the attendance file
        path = os.path.join(downloads_directory(), f"{name_of_file}.xlsx")
        workbook = xlsxwriter.Workbook(path)

        worksheet = workbook.add_worksheet()

        # Fix: the date mixed English and Spanish ("Date of attendance: miércoles 18 de noviembre del 2026")
        first_line = tiempo.strftime("Date of attendance: %Y-%m-%d %H:%M:%S")

        #Writing the first line and the columns of the documents
        worksheet.write(0,0,first_line)
        worksheet.write(2,0,"Id")
        worksheet.write(2, 1, "Last Names(s)")
        worksheet.write(2, 2, "Name(s)")

        # Fix: clear the rows of the previous attendance before showing the new one
        self.tree.delete(*self.tree.get_children())

        row = 4
        for student in data:
            self.tree.insert("", "end", values=(student[0],student[1],student[2]))
            worksheet.write(row,0, student[0])
            worksheet.write(row, 1, student[1])
            worksheet.write(row, 2, student[2])
            row += 1

        self.tree.pack()
        # Fix: the file is written (closed) before telling the user it's ready
        workbook.close()
        messagebox.showinfo("",
                            f"{len(data)} student(s) present.\n"
                            f"The list of attendance was saved in:\n{path}")


class Register(tk.Frame):

    def __init__(self,master):
        tk.Frame.__init__(self,master)
        self.master = master
        tk.Frame.config(self,cnf=None, bg="#264653",pady=50)
        font_example = tkFont.Font(family="Cambria", size=18, weight="bold", slant="italic")

        # Fix: StringVar instead of IntVar. Clearing an IntVar with "" (or typing a
        # letter) made .get() crash; now the number is validated in add_student.
        self.no_cuenta = tk.StringVar()
        self.name = tk.StringVar()
        self.last_name = tk.StringVar()

        self.conexion = connect()
        self.cursor = self.conexion.cursor()

        tk.Label(self, text="No. Cuenta",bg="#264653",fg="white", font=font_example).pack(pady=10)
        tk.Entry(self, justify="center",
                 font='Times 16 italic',
                 textvariable=self.no_cuenta,
                 bg="white").pack(pady=10)

        tk.Label(self, text="Name(s):",bg="#264653", fg="white", font=font_example).pack(pady=10)
        tk.Entry(self, justify="center",
                 font='Times 16 italic',
                 textvariable=self.name,
                 bg="white").pack(pady=10)

        tk.Label(self, text="Last Name(s):", bg="#264653",fg="white",  font=font_example).pack(pady=10)
        tk.Entry(self, justify="center",
                 font='Times 16 italic',
                 textvariable=self.last_name,
                 bg="white").pack(pady=10)

        tk.Button(self, text="Add Student",
                  bg="#faedcd",
                  command= self.add_student,
                  font= 'Times 16 italic',
                  cursor="hand2").pack(pady=20)

        # Fix: "Back" only goes back. Before, it was the only place where the model
        # was trained, so closing the window instead lost the new students.
        tk.Button(self, text="<- Back", bg="#f3722c",
                  fg="white" ,font='Times 14 italic',
                  cursor='hand2',
                  command=lambda: master.switch_frames(Start)).pack(side="left",pady=20)


    def add_student(self):
        # Fix: validate the data before saving anything
        no_cuenta = self.no_cuenta.get().strip()
        name = self.name.get().strip()
        last_name = self.last_name.get().strip()

        if not no_cuenta.isdigit():
            messagebox.showerror("", "No. Cuenta must be a number.")
            return
        if not name or not last_name:
            messagebox.showerror("", "Write the name(s) and last name(s).")
            return

        self.cursor.execute("SELECT 1 FROM Students WHERE id = ?", (int(no_cuenta),))
        if self.cursor.fetchone():
            messagebox.showerror("", f"There is already a student with No. Cuenta {no_cuenta}.")
            return

        # Fix: take the pictures first and save the student only if it worked.
        # Before, the student was saved even if the camera failed.
        add_student_photos = DatasetFaces()
        pictures = add_student_photos.start(int(no_cuenta))
        if pictures == 0:
            messagebox.showerror("", "No pictures were taken. The student was not added.")
            return

        self.cursor.execute("INSERT INTO Students VALUES (?,?,?)",
                            (int(no_cuenta), last_name, name))
        self.conexion.commit()

        # Fix: train right after adding the student, so they can be recognized
        self.train_model()
        messagebox.showinfo("", f"Student Added Successfully ({pictures} pictures).")
        self.clean_fields()

    def train_model(self):
        trainer = Trainer()
        trainer.train()


    def clean_fields(self):
        self.no_cuenta.set("")
        self.name.set("")
        self.last_name.set("")
