import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk

class FixedAspectCropper:
    def __init__(self, master):
        self.master = master
        master.title("Fixed Aspect Ratio Crop Tool")

        # Aspect ratios: 48″×8.5″ or 48″×17″
        self.ratio_short = 48.0 / 8.5
        self.ratio_tall  = 48.0 / 17.0
        self.aspect_ratio = self.ratio_short

        # Canvas for image & cropping
        self.canvas = tk.Canvas(master, cursor="cross")
        self.canvas.pack(fill=tk.BOTH, expand=True)

        # Controls
        ctrl_frame = tk.Frame(master)
        ctrl_frame.pack(fill=tk.X)
        tk.Button(ctrl_frame, text="Load Image",   command=self.load_image).pack(side=tk.LEFT)
        self.save_btn = tk.Button(ctrl_frame, text="Save Crop", command=self.save_crop, state=tk.DISABLED)
        self.save_btn.pack(side=tk.LEFT)
        # Checkbox to switch height
        self.tall_var = tk.BooleanVar(value=False)
        tk.Checkbutton(
            ctrl_frame,
            text='17″ tall',
            variable=self.tall_var,
            command=self.update_aspect_ratio
        ).pack(side=tk.LEFT, padx=10)

        # Bindings
        self.canvas.bind("<ButtonPress-1>",   self.on_button_press)
        self.canvas.bind("<B1-Motion>",       self.on_mouse_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_button_release)
        master.bind_all("<Key>", self.on_key_press)

        # State
        self.image       = None
        self.tkimage     = None
        self.scale       = 1.0
        self.rect_id     = None
        self.start_x     = None
        self.start_y     = None
        self.crop_coords = None

    def update_aspect_ratio(self):
        """Toggle between short and tall aspect ratios."""
        self.aspect_ratio = self.ratio_tall if self.tall_var.get() else self.ratio_short
        # Optionally adjust existing rectangle to new ratio:
        if self.rect_id:
            self._scale_rect(1.0)  # re-center/rescale to maintain ratio

    def load_image(self):
        path = filedialog.askopenfilename(
            filetypes=[("Image files","*.jpg;*.jpeg;*.png;*.bmp;*.tif;*.tiff")]
        )
        if not path:
            return
        self.image = Image.open(path)
        self._redraw_image()
        self.save_btn.config(state=tk.DISABLED)

    def _redraw_image(self):
        w, h = self.image.size
        max_w = self.master.winfo_screenwidth() - 100
        max_h = self.master.winfo_screenheight() - 200
        ratio = min(max_w / w, max_h / h, 1.0)
        self.scale = ratio

        disp = self.image.resize((int(w * ratio), int(h * ratio)), Image.LANCZOS)
        self.tkimage = ImageTk.PhotoImage(disp)
        self.canvas.config(width=disp.width, height=disp.height)
        self.canvas.create_image(0, 0, image=self.tkimage, anchor="nw")
        if self.rect_id:
            self.canvas.delete(self.rect_id)
            self.rect_id = None

    def on_button_press(self, event):
        if not self.image:
            return
        self.start_x, self.start_y = event.x, event.y
        if self.rect_id:
            self.canvas.delete(self.rect_id)
            self.rect_id = None

    def on_mouse_drag(self, event):
        if self.start_x is None or self.start_y is None:
            return
        dx = event.x - self.start_x
        dir_x = 1 if dx >= 0 else -1
        width = abs(dx)
        height = width / self.aspect_ratio
        dy_event = event.y - self.start_y
        dir_y = 1 if dy_event >= 0 else -1

        x2 = self.start_x + dir_x * width
        y2 = self.start_y + dir_y * height

        if self.rect_id:
            self.canvas.coords(self.rect_id, self.start_x, self.start_y, x2, y2)
        else:
            self.rect_id = self.canvas.create_rectangle(
                self.start_x, self.start_y, x2, y2, outline="red", width=2
            )

    def on_button_release(self, event):
        if not self.rect_id:
            return
        self.crop_coords = self.canvas.coords(self.rect_id)
        self.save_btn.config(state=tk.NORMAL)

    def on_key_press(self, event):
        if not self.rect_id:
            return
        ctrl = (event.state & 0x4) != 0
        # Scale: +/= , -/_ 
        if event.keysym in ("plus","equal"):
            factor = 1.1 if not ctrl else 1.02
            self._scale_rect(factor)
        elif event.keysym in ("minus","underscore"):
            factor = 0.9 if not ctrl else 0.98
            self._scale_rect(factor)
        # Move: arrows
        elif event.keysym in ("Left","Right","Up","Down"):
            step = 10 if not ctrl else 1
            dx = step if event.keysym=="Right" else -step if event.keysym=="Left" else 0
            dy = step if event.keysym=="Down"  else -step if event.keysym=="Up"   else 0
            self._move_rect(dx, dy)

    def _scale_rect(self, factor):
        x1,y1,x2,y2 = self.canvas.coords(self.rect_id)
        # enforce aspect ratio on height
        cx, cy = (x1+x2)/2, (y1+y2)/2
        w = (x2-x1)*factor
        h = w / self.aspect_ratio
        x1n, y1n = cx - w/2, cy - h/2
        x2n, y2n = cx + w/2, cy + h/2
        self.canvas.coords(self.rect_id, x1n,y1n,x2n,y2n)
        self.crop_coords = [x1n,y1n,x2n,y2n]

    def _move_rect(self, dx, dy):
        x1,y1,x2,y2 = self.canvas.coords(self.rect_id)
        x1n, y1n, x2n, y2n = x1+dx, y1+dy, x2+dx, y2+dy
        # Clamp to canvas bounds
        W,H = self.canvas.winfo_width(), self.canvas.winfo_height()
        if x1n<0:       x2n-=x1n; x1n=0
        if y1n<0:       y2n-=y1n; y1n=0
        if x2n> W: x1n-=x2n-W; x2n=W
        if y2n> H: y1n-=y2n-H; y2n=H
        self.canvas.coords(self.rect_id, x1n,y1n,x2n,y2n)
        self.crop_coords = [x1n,y1n,x2n,y2n]

    def save_crop(self):
        if not self.crop_coords or not self.image:
            return
        x1,y1,x2,y2 = self.crop_coords
        left  = int(min(x1,x2)/self.scale)
        upper = int(min(y1,y2)/self.scale)
        right = int(max(x1,x2)/self.scale)
        lower = int(max(y1,y2)/self.scale)
        cropped = self.image.crop((left,upper,right,lower))

        save_path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG","*.png"),("JPEG","*.jpg;*.jpeg")]
        )
        if not save_path:
            return
        cropped.save(save_path)
        messagebox.showinfo("Saved", f"Cropped image saved to:\n{save_path}")

if __name__ == "__main__":
    # pip install pillow
    root = tk.Tk()
    app = FixedAspectCropper(root)
    root.mainloop()
