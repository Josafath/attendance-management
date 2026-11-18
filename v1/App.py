import tkinter.font as tkFont
import sqlite3
from tkinter import ttk, messagebox
import xlsxwriter
import locale, datetime, os
import tkinter as tk

from src.Recognition_files.datasetFaces import DatasetFaces
from src.Recognition_files.trainerFaces import Trainer
from src.Recognition_files.recognitionFaces import Recognition

class Start(tk.Frame):

    def __init__(self, master):
        locale.setlocale(locale.LC_ALL, '')
        self.tiempo = datetime.datetime.now()

        tk.Frame.__init__(self, master)
        tk.Frame.config(self,cnf=None, bg="#264653")

        self.conexion = sqlite3.connect("Students.db")
        self.cursor = self.conexion.cursor()

        tk.Button(self, text="Register Student",
                  font= ('Times 18 bold'),relief='raised',
                  bg="#f1faee",activebackground="#ffe8d6",
                  command=lambda: master.switch_frames(Register),cursor="hand2").pack(pady=20)

        tk.Button(self, text="Take Attendance",
                  font=('Times 18 bold'),relief='raised',
                  bg="#f1faee",activebackground="#ffe8d6",
                  command=self.recognition,cursor="hand2").pack(pady=20)

        self.tree = ttk.Treeview(master, columns=('Id', 'Last Name(s)', 'Name(s)'), show='headings')
        cols = ('Id', 'Last Name(s)', 'Name(s)')
        for col in cols:
            self.tree.heading(col, text=col)

    def recognition(self):
        recognition = Recognition()
        ids_students_in_class = recognition.go()
        self.cursor.execute(f"SELECT * FROM Students "
                            f"WHERE id "
                            f"IN ('{','.join(str(student) for student in ids_students_in_class)}')")

        #Getting the data from query
        data = self.cursor.fetchall()

        name_of_file = "Attendance"+str(self.tiempo.strftime("%d_%B_%Y"))

        #Setting path for the attendance file
        workbook = xlsxwriter.Workbook(f"{os.path.expanduser('~')}/Downloads/{name_of_file}.xlsx")

        worksheet = workbook.add_worksheet()

        first_line = self.tiempo.strftime("Date of attendance: %A %d de %B del %Y a las %I:%M:%S")

        #Writing the first line and the columns of the documents
        worksheet.write(0,0,first_line)
        worksheet.write(2,0,"Id")
        worksheet.write(2, 1, "Last Names(s)")
        worksheet.write(2, 2, "Name(s)")

        row = 4
        for student in data:
            self.tree.insert("", "end", values=(student[0],student[1],student[2]))
            worksheet.write(row,0, student[0])
            worksheet.write(row, 1, student[1])
            worksheet.write(row, 2, student[2])
            row += 1

        self.tree.pack()
        messagebox.showinfo("",
                            "The list of attendance has been downloaded and it's on the Downloads Directory")
        workbook.close()
        self.conexion.commit()


class Register(tk.Frame):

    def __init__(self,master):
        tk.Frame.__init__(self,master)
        self.master = master
        tk.Frame.config(self,cnf=None, bg="#264653",pady=50)
        font_example = tkFont.Font(family="Cambria", size=18, weight="bold", slant="italic")

        self.no_cuenta = tk.IntVar()
        self.name = tk.StringVar()
        self.last_name = tk.StringVar()

        self.conexion = sqlite3.connect("Students.db")
        self.cursor = self.conexion.cursor()

        self.cursor.execute("CREATE TABLE IF NOT EXISTS Students "
                            "(id INT PRIMARY KEY,"
                            " last_name VARCHAR(50),"
                            " name VARCHAR(50))")

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

        tk.Button(self, text="<- Back", bg="#f3722c",
                  fg="white" ,font='Times 14 italic',
                  cursor='hand2',
                  command=self.train_model).pack(side="left",pady=20)


    def add_student(self):

        student_data = [(self.no_cuenta.get(), self.last_name.get(), self.name.get())]
        self.cursor.executemany("INSERT INTO Students VALUES (?,?,?)", student_data)

        add_student_photos = DatasetFaces()
        add_student_photos.start(self.no_cuenta.get())
        self.cursor.fetchall()
        self.conexion.commit()
        self.clean_fields()

    def train_model(self):
        trainer = Trainer()
        trainer.train()
        self.master.switch_frames(Start)


    def clean_fields(self):
        self.no_cuenta.set("")
        self.name.set("")
        self.last_name.set("")


