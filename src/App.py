import tkinter as tk
import tkinter.font as tkFont

class Start(tk.Frame):

    def __init__(self, master):
        tk.Frame.__init__(self, master)
        tk.Frame.config(self,cnf=None, bg="#264653")

        tk.Button(self, text="Register Student", font= ('Helvetica 25 bold'),relief='raised',bg="#f1faee",activebackground="#ffe8d6",command=lambda: master.switch_frames(Register),cursor="hand2").pack(pady=50)
        tk.Button(self, text="Take Attendance", font=('Helvetica 25 bold'),relief='raised',bg="#f1faee",activebackground="#ffe8d6",cursor="hand2").pack(pady=50)



class Register(tk.Frame):

    def __init__(self,master):
        tk.Frame.__init__(self,master)
        tk.Frame.config(self,cnf=None, bg="#264653",pady=50)
        font_example = tkFont.Font(family="Cambria", size=18, weight="bold", slant="italic")

        self.no_cuenta = tk.IntVar()
        self.name = tk.StringVar()
        self.last_name = tk.StringVar()

        tk.Label(self, text="No. Cuenta",bg="#264653",fg="white", font=font_example).pack(pady=10)
        tk.Entry(self, justify="center", font='Times 16 italic',textvariable=self.no_cuenta, bg="white").pack(pady=10)

        tk.Label(self, text="Name(s):",bg="#264653", fg="white", font=font_example).pack(pady=10)
        tk.Entry(self, justify="center", font='Times 16 italic',textvariable=self.no_cuenta, bg="white").pack(pady=10)

        tk.Label(self, text="Last Name:", bg="#264653",fg="white",  font=font_example).pack(pady=10)
        tk.Entry(self, justify="center", font='Times 16 italic',textvariable=self.no_cuenta, bg="white").pack(pady=10)

        tk.Button(self, text="Add Student", bg="#faedcd", font= 'Times 16 italic', cursor="hand2").pack(pady=20)
        tk.Button(self, text="<- Back", bg="#f3722c", fg="white" ,font='Times 14 italic', cursor='hand2').pack(side="left",pady=20)
