import tkinter as tk
from tkinter import ttk
from test_hand_lendmarks.callable import ASLDetector
from PIL import ImageTk, Image


class ASL_GUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Text App")

        self.switch_var = tk.BooleanVar(value=False)
        self.switch_var.trace('w', self.toggle_switch)
        self.switch = ttk.Checkbutton(root, text="Toggle Image", variable=self.switch_var)
        self.switch.pack(pady=10)

        self.update_button = ttk.Button(root, text="Update Image", command=self.update)
        self.update_button.pack(pady=10)

        self.frame = ttk.Frame(root)
        self.frame.pack(fill=tk.BOTH, expand=True)

        self.image_label = ttk.Label(self.frame)
        self.image_label.pack(side=tk.LEFT, padx=10, pady=10)

        self.text_box = tk.Text(self.frame, wrap="word")
        self.text_box.pack(side=tk.LEFT, padx=10, pady=10, fill=tk.BOTH, expand=True)

        self.raw_text = ""

    def toggle_switch(self, *args):
        if self.switch_var.get():
            self.text_box.config(state='normal')
        else:
            self.text_box.configure(state='disabled')

    def update_text(self):
        self.text_box.config(state='normal')
        self.text_box.delete(1.0, tk.END)
        self.text_box.insert(tk.END, self.raw_text)
        self.text_box.configure(state='disabled')

    def update(self, image=None, letter=None):
        if image is not None:
            image = Image.fromarray(image)
            photo = ImageTk.PhotoImage(image)
            self.image_label.configure(image=photo)
            self.image_label.image = photo
        if letter is not None:
            if letter == " ":
                if self.raw_text != "":
                    if self.raw_text[-1] != " ":
                        self.raw_text += letter
            else:
                if self.raw_text != "":
                    if self.raw_text[-1] != letter:
                        self.raw_text += letter
                else:
                    self.raw_text += letter

            self.update_text()


root = tk.Tk()
app = ASL_GUI(root)

my_object = ASLDetector(app.update)
my_object.start_loop()

root.mainloop()

my_object.stop_loop()
