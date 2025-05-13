import cv2
import os
import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk
import glob
import re

# --- Configuration ---
INPUT_DIR = r"C:\Users\nickh\OneDrive\Documents\nh\Git_repo\coin_sorter3\data\obverse"
OUTPUT_DIR = r"C:\Users\nickh\OneDrive\Documents\nh\Git_repo\coin_sorter3\data\date"
DISPLAY_WIDTH = 900   # fixed display width in pixels
DISPLAY_HEIGHT = 900  # fixed display height in pixels

os.makedirs(OUTPUT_DIR, exist_ok=True)

class ROIExtractor(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Draw ROI and Press Enter")

        self.canvas = tk.Canvas(self, width=DISPLAY_WIDTH, height=DISPLAY_HEIGHT, cursor="cross")
        self.canvas.pack(fill=tk.BOTH, expand=True)

        self.bind("<Return>", self.save_crop)
        self.bind("<Left>", self.go_back)
        self.bind("<Right>", self.skip_forward)

        self.canvas.bind("<Button-1>", self.start_crop)
        self.canvas.bind("<B1-Motion>", self.update_crop)

        self.files = sorted(glob.glob(os.path.join(INPUT_DIR, '*.jpg')) + glob.glob(os.path.join(INPUT_DIR, '*.png')))
        self.index = 0
        self.roi_rect = None

        self.exit_button = tk.Button(self, text="Exit", command=self.quit)
        self.exit_button.pack(side=tk.BOTTOM, fill=tk.X)

        self.load_image()

    def load_image(self):
        if self.index >= len(self.files):
            messagebox.showinfo("Done", "All images processed.")
            self.quit()
            return

        self.image_path = self.files[self.index]
        self.original = cv2.imread(self.image_path)
        h, w = self.original.shape[:2]
        scale = min(DISPLAY_WIDTH / w, DISPLAY_HEIGHT / h)
        self.display_scale = scale

        self.display_img = cv2.resize(self.original, (int(w * scale), int(h * scale)))
        self.tk_img = ImageTk.PhotoImage(image=Image.fromarray(cv2.cvtColor(self.display_img, cv2.COLOR_BGR2RGB)))

        self.canvas.delete("all")
        self.canvas.config(width=self.tk_img.width(), height=self.tk_img.height())
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.tk_img)
        self.roi_rect = None

    def start_crop(self, event):
        self.start_x = event.x
        self.start_y = event.y
        self.roi_rect = self.canvas.create_rectangle(self.start_x, self.start_y, self.start_x, self.start_y, outline='yellow', width=2)

    def update_crop(self, event):
        if self.roi_rect:
            self.canvas.coords(self.roi_rect, self.start_x, self.start_y, event.x, event.y)

    def save_crop(self, event=None):
        if not self.roi_rect:
            return

        x1, y1, x2, y2 = [int(coord / self.display_scale) for coord in self.canvas.coords(self.roi_rect)]
        x1, x2 = sorted((max(0, x1), min(self.original.shape[1], x2)))
        y1, y2 = sorted((max(0, y1), min(self.original.shape[0], y2)))

        crop = self.original[y1:y2, x1:x2]
        if crop.size == 0:
            messagebox.showerror("Error", "Invalid crop region")
            return

        base = os.path.basename(self.image_path)
        name, ext = os.path.splitext(base)
        new_name = re.sub(r"(.*_)(\d{4})$", r"\g<1>1\2", name) + ext
        out_path = os.path.join(OUTPUT_DIR, new_name)

        if os.path.exists(out_path):
            if not messagebox.askyesno("Overwrite?", f"{new_name} exists. Overwrite?"):
                self.index += 1
                self.load_image()
                return

        cv2.imwrite(out_path, crop)
        self.index += 1
        self.load_image()

    def skip_forward(self, event=None):
        self.index += 1
        if self.index >= len(self.files):
            messagebox.showinfo("Done", "All images processed.")
            self.quit()
        else:
            self.load_image()

    def go_back(self, event=None):
        if self.index > 0:
            self.index -= 1
            self.load_image()

if __name__ == '__main__':
    app = ROIExtractor()
    app.mainloop()
