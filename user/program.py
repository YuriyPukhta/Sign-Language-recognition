import threading
import tkinter as tk
from tkinter import ttk
import numpy as np
from PIL import ImageTk, Image
from autocorrect import Speller
import cv2
from test_hand_lendmarks.callable import ASLDetector
from test_image.callable import ResNetDetector

frames_skipped = 3

def fix_spelling(text):
    spell = Speller(lang='en')
    words = text.split()
    corrected_words = [spell(word) for word in words]
    return ' '.join(corrected_words)


class VideoCapture:
    def __init__(self, callback):
        self.cap = cv2.VideoCapture(0)
        self.callback = callback
        self.running = False
        self.thread = None

    def start_loop(self):
        self.running = True
        self.thread = threading.Thread(target=self.loop)
        self.thread.start()

    def stop_loop(self):
        self.running = False
        self.thread.join()

    def loop(self):
        while self.running:
            for _ in range(frames_skipped+1):
                _, frame = self.cap.read()
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self.callback(frame_rgb)


class ASL_GUI:
    def __init__(self, root):
        self.CNNDetector = ASLDetector()
        self.ResNetDetector = ResNetDetector()

        self.selected_model = 0

        self.root = root
        self.root.title("Image Text App")

        # Create a dropdown menu and a button at the top
        self.option_var = tk.StringVar(value="CNN")
        self.option_var.trace('w', self.select_option)
        self.option_combobox = ttk.Combobox(root, textvariable=self.option_var, values=["CNN", "ResNet"])
        self.option_combobox.pack(pady=10)

        # self.update_button = ttk.Button(root, text="Update Image", command=self.update_image)
        # self.update_button.pack(pady=10)

        # Create a frame to hold the image and text boxes
        self.frame = ttk.Frame(root)
        self.frame.pack(fill=tk.BOTH, expand=True)

        # Create the image label on the left side
        self.image_label = ttk.Label(self.frame)
        self.image_label.grid(row=0, column=0, padx=10, pady=10, rowspan=3)

        # Create the text boxes on the right side
        self.text_box1 = tk.Text(self.frame, wrap="word")
        self.text_box1.grid(row=0, column=1, padx=10, pady=10, sticky="nsew")
        self.text_box2 = tk.Text(self.frame, wrap="word")
        self.text_box2.grid(row=1, column=1, padx=10, pady=10, sticky="nsew")
        self.text_box3 = tk.Text(self.frame, wrap="word")
        self.text_box3.grid(row=2, column=1, padx=10, pady=10, sticky="nsew")

        # Configure grid weights for resizing
        self.frame.columnconfigure(1, weight=1)
        self.frame.rowconfigure(0, weight=1)
        self.frame.rowconfigure(1, weight=1)
        self.frame.rowconfigure(2, weight=1)

        self.raw_text = ""
        self.merged_text = ""
        self.spelling_fixed_text = ""

        self.buffer = ""

        self.update()

    def select_option(self, *args):
        option = self.option_var.get()
        if option == "CNN":
            self.selected_model = 0
        elif option == "ResNet":
            self.selected_model = 1

    def update_text(self):
        self.text_box1.configure(state='normal')
        self.text_box2.configure(state='normal')
        self.text_box3.configure(state='normal')

        self.text_box1.delete(1.0, tk.END)
        self.text_box2.delete(1.0, tk.END)
        self.text_box3.delete(1.0, tk.END)

        self.text_box1.insert(tk.END, fix_spelling(self.merged_text))
        self.text_box2.insert(tk.END, self.merged_text)
        self.text_box3.insert(tk.END, self.raw_text)

        self.text_box1.configure(state='disabled')
        self.text_box2.configure(state='disabled')
        self.text_box3.configure(state='disabled')

    def update_img(self, image):
        image = Image.fromarray(image)
        photo = ImageTk.PhotoImage(image)
        self.image_label.configure(image=photo)
        self.image_label.image = photo

    def update(self, image=None):
        if image is not None:
            if self.selected_model == 0:
                letter = self.CNNDetector.get_letter(image)
            elif self.selected_model == 1:
                letter = self.ResNetDetector.get_letter(image)

            print(letter)

            if letter is not None:
                if self.merged_text != "":
                    if self.merged_text[-1] != " ":
                        self.raw_text += letter

                if letter == " ":
                    if self.merged_text != "":
                        if self.merged_text[-1] != " ":
                            self.merged_text += letter
                            self.update_text()
                else:
                    if self.merged_text != "":
                        if self.merged_text[-1] != letter:
                            self.merged_text += letter
                    else:
                        self.merged_text += letter
                    self.update_text()
        else:
            image = np.ones((480, 640))
            self.update_text()

        self.update_img(image)


root = tk.Tk()
app = ASL_GUI(root)

video_cap = VideoCapture(app.update)
video_cap.start_loop()

app_width = 1300
app_height = 600
root.geometry(f"{app_width}x{app_height}")

root.mainloop()

video_cap.stop_loop()
