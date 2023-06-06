import tkinter as tk


root = tk.Tk()
root.geometry("400x400")
class MenuFrame:
    def __int__(self, root):
        self.menu_frame = tk.Frame(root, bg="white")
        num_rows = 5
        num_columns = 3
        for row in range(num_rows):
            self.menu_frame.grid_rowconfigure(row, weight=1)

        for column in range(num_columns):
            self.menu_frame.grid_columnconfigure(column, weight=1)

        self.menu_frame.grid(row=0, column=0, sticky="nsew")

        button1 = tk.Button(self.menu_frame, text="Start", highlightthickness=5, bd=0)
        button1.grid(row=1, column=1)

        button2 = tk.Button(self.menu_frame, text="Info", highlightthickness=5, bd=0)
        button2.grid(row=2, column=1)

        button3 = tk.Button(self.menu_frame, text="Exit", highlightthickness=0, bd=0)
        button3.grid(row=3, column=1)
#root.resizable(False, False)
# Отримання розмірів екрану
screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()

# Визначення розташування кнопок по центру









# Розташування фрейма по центру екрану# Оновити вікно перед отриманням розмірів

root.grid_rowconfigure(0, weight=1)  # Розтягувати рядок 0
root.grid_columnconfigure(0, weight=1)
root.mainloop()
