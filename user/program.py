import threading
import tkinter as tk
from tkinter import ttk
import numpy as np
from PIL import ImageTk, Image
from autocorrect import Speller
import cv2
from test_hand_lendmarks.callable import ASLDetector
from test_image.callable import ResNetDetector
from scipy.stats import zscore
from collections import Counter

frames_skipped = 3
zscore_treshold = 1.0
percent_treshold = 0.4
enable_zscore_check = True
enable_percent_check = False
count_treshold = 10


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
            for _ in range(frames_skipped + 1):
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
        self.root.configure(background="#2b2d30")

        self.frame = ttk.Frame(root, style="My.TFrame")
        self.frame.pack(fill=tk.BOTH, expand=True)
        self.root.style = ttk.Style()
        self.root.style.configure("My.TFrame", background="#2b2d30", borderwidth=0)

        self.combobox_style = ttk.Style()
        self.combobox_style.configure("TCombobox", background="gray", relief="flat")
        self.option_var = tk.StringVar(value="CNN")
        self.option_var.trace('w', self.select_option)
        self.option_combobox = ttk.Combobox(self.frame, textvariable=self.option_var, values=["CNN", "ResNet"], style="TCombobox")
        self.option_combobox.grid(row=0, column=0, padx=10, pady=10)

        self.image_label = ttk.Label(self.frame, style="NoBorder.TLabel", background="#2b2d30")
        self.image_label.grid(row=1, column=0, padx=10, pady=0, rowspan=23)

        self.label1 = ttk.Label(self.frame, text="Fixed Spelling", background="#2b2d30", foreground="white", font=("SegoeUI", 12))
        self.label1.grid(row=0, column=1, padx=10, pady=1, sticky="w")

        self.text_box1 = tk.Text(self.frame, wrap="word", bg="#1e1f22", fg="white", font=("SegoeUI", 12), highlightthickness=0, bd=0)
        self.text_box1.grid(row=1, column=1, padx=10, pady=10, rowspan=7)

        self.label2 = ttk.Label(self.frame, text="Stacked", background="#2b2d30", foreground="white", font=("SegoeUI", 12))
        self.label2.grid(row=8, column=1, padx=10, pady=1, sticky="w")

        self.text_box2 = tk.Text(self.frame, wrap="word", bg="#1e1f22", fg="white", font=("SegoeUI", 12), highlightthickness=0, bd=0)
        self.text_box2.grid(row=9, column=1, padx=10, pady=10, rowspan=7)

        self.label2 = ttk.Label(self.frame, text="Raw buffer", background="#2b2d30", foreground="white", font=("SegoeUI", 12))
        self.label2.grid(row=16, column=1, padx=10, pady=1, sticky="w")

        self.text_box3 = tk.Text(self.frame, wrap="word", bg="#1e1f22", fg="white", font=("SegoeUI", 12), highlightthickness=0, bd=0)
        self.text_box3.grid(row=17, column=1, padx=10, pady=10, rowspan=7)

        self.frame.columnconfigure(1, weight=1)
        self.frame.rowconfigure(1, weight=1)
        self.frame.rowconfigure(9, weight=1)
        self.frame.rowconfigure(17, weight=1)

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

    def update_text(self, only_buffer=False):
        if not only_buffer:
            self.text_box1.configure(state='normal')
            self.text_box2.configure(state='normal')
            self.text_box1.delete(1.0, tk.END)
            self.text_box2.delete(1.0, tk.END)
            self.text_box1.insert(tk.END, fix_spelling(self.merged_text))
            self.text_box2.insert(tk.END, self.merged_text)
            self.text_box1.configure(state='disabled')
            self.text_box2.configure(state='disabled')

        self.text_box3.configure(state='normal')
        self.text_box3.delete(1.0, tk.END)
        self.text_box3.insert(tk.END, self.buffer)
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

            if letter is not None:
                if letter == " ":
                    if self.merged_text != "":
                        if self.merged_text[-1] != " ":
                            self.merged_text += letter
                            self.update_text()
                        self.buffer = ""
                else:
                    self.buffer += letter
                    counts = list(Counter(self.buffer).values())
                    z_scores = zscore(counts)
                    max_zscore_idx = np.argmax(z_scores)
                    p_scores = np.array(counts) / sum(counts)
                    z_score = np.nan_to_num(z_scores[max_zscore_idx])
                    p_score = p_scores[max_zscore_idx]

                    print(z_score, p_score)

                    if ((z_score > zscore_treshold or z_score == 0.0 or not enable_zscore_check) and (
                            p_score > percent_treshold or not enable_percent_check) and len(self.buffer) > 7) or len(
                        self.buffer) > 25:
                        letter = str(list(Counter(self.buffer).keys())[max_zscore_idx])
                        self.buffer = ""

                        print("letter: ", letter)
                        if self.merged_text != "":
                            if self.merged_text[-1] != letter:
                                self.merged_text += letter
                                self.update_text()
                            elif p_score > 0.5:
                                self.merged_text += letter
                                self.update_text()
                        else:
                            self.merged_text += letter
                            self.update_text()
                        return

                    self.update_text(True)
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
