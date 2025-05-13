import cv2
import numpy as np
import os
import glob
import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
import re
from functools import cmp_to_key

# --- Configuration ---
BASE_DIR = r"C:\Users\nickh\OneDrive\Documents\nh\Git_repo\coin_sorter3"
INPUT_DIRS = {
    'liberty': os.path.join(BASE_DIR, "data", "liberty"),
    'date': os.path.join(BASE_DIR, "data", "date"),
}
PREVIEW_SIZE = 600  # Size for the image preview
ZOOM_FACTOR = 2.5   # How much to zoom in when viewing cropped regions

class ROIRecropperApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Penny ROI Recropper")
        self.geometry("1400x900")
        self.configure(bg='#f0f0f0')
        
        self.create_ui()
        self.mode = None
        self.files = []
        self.idx = 0
        self.img = None
        self.h, self.w = 0, 0
        self.roi_rect = None
        self.display_scale = 1.0
        self.original_img = None
        
        # Start screen
        self.show_start_screen()

    def create_ui(self):
        # Main frame layout
        self.main_frame = ttk.Frame(self)
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Image frame (left side)
        self.img_frame = ttk.Frame(self.main_frame)
        self.img_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Canvas for image display
        self.canvas = tk.Canvas(self.img_frame, bg='black')
        self.canvas.pack(fill=tk.BOTH, expand=True)
        
        # Bind mouse events for drawing ROI
        self.canvas.bind("<ButtonPress-1>", self.on_mouse_down)
        self.canvas.bind("<B1-Motion>", self.on_mouse_move)
        self.canvas.bind("<ButtonRelease-1>", self.on_mouse_up)
        
        # Variables for ROI drawing
        self.start_x = None
        self.start_y = None
        self.drawing = False
        self.rect_id = None
        
        # Control frame (right side)
        self.ctrl_frame = ttk.Frame(self.main_frame, width=300)
        self.ctrl_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=(10, 0))
        
        # Create a start screen frame
        self.start_frame = ttk.Frame(self.main_frame)
        
        # Status bar
        self.status_var = tk.StringVar(value="Ready")
        self.status_bar = ttk.Label(self, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)

    def show_start_screen(self):
        # Clear any existing widgets
        for widget in self.ctrl_frame.winfo_children():
            widget.destroy()
        
        self.canvas.delete("all")
        
        # Title
        ttk.Label(self.ctrl_frame, text="ROI Recropper", 
                  font=("Arial", 16, "bold")).pack(pady=(0, 20))
        
        # Mode selection buttons
        ttk.Label(self.ctrl_frame, text="Select ROI Type:", 
                  font=("Arial", 12)).pack(pady=(10, 5))
        
        # Large buttons for each task
        button_style = {"width": 20, "padding": 10}
        
        btn_liberty = ttk.Button(self.ctrl_frame, text="Recrop Liberty ROIs", 
                               command=lambda: self.start_task('liberty'), **button_style)
        btn_liberty.pack(pady=5)
        
        btn_date = ttk.Button(self.ctrl_frame, text="Recrop Date ROIs", 
                            command=lambda: self.start_task('date'), **button_style)
        btn_date.pack(pady=5)
        
        # Exit button
        ttk.Button(self.ctrl_frame, text="Exit", 
                 command=self.quit, **button_style).pack(pady=(20, 5))

    def get_files_by_size(self, directory):
        """Get files from directory sorted by size (largest first)"""
        file_list = []
        
        # Get all image files in the directory
        file_paths = glob.glob(os.path.join(directory, "*.png")) + glob.glob(os.path.join(directory, "*.jpg"))
        
        # Add size information
        for file_path in file_paths:
            file_size = os.path.getsize(file_path)
            file_list.append((file_path, file_size))
        
        # Sort by size (largest first)
        file_list.sort(key=lambda x: x[1], reverse=True)
        
        # Return just the file paths
        return [f[0] for f in file_list]

    def start_task(self, roi_type):
        self.mode = roi_type
        
        # Get files sorted by size
        self.files = self.get_files_by_size(INPUT_DIRS[roi_type])
        
        if not self.files:
            messagebox.showinfo("No Files", f"No image files found in {INPUT_DIRS[roi_type]}")
            return
        
        self.idx = 0
        self.show_cropper()
        self.load_image()

    def show_cropper(self):
        # Clear existing controls
        for widget in self.ctrl_frame.winfo_children():
            widget.destroy()
        
        # Title
        ttk.Label(self.ctrl_frame, text=f"Recrop {self.mode.capitalize()} ROIs", 
                  font=("Arial", 14, "bold")).pack(pady=(0, 20))
        
        # File info
        self.file_info_var = tk.StringVar(value="")
        ttk.Label(self.ctrl_frame, textvariable=self.file_info_var, 
                 font=("Arial", 10)).pack(pady=5)
        
        # ROI adjustment instructions
        instruction_text = """
        Controls:
        
        Click and drag: Draw ROI box
        Arrow keys: Fine-tune position
        +/- keys: Resize ROI
        Enter: Save and continue
        """
        
        ttk.Label(self.ctrl_frame, text=instruction_text, 
                 justify=tk.LEFT).pack(pady=10, padx=10)
        
        # Navigation buttons
        nav_frame = ttk.Frame(self.ctrl_frame)
        nav_frame.pack(pady=20, fill=tk.X)
        
        ttk.Button(nav_frame, text="Previous", command=self.prev_image, 
                  width=12).pack(side=tk.LEFT, padx=5)
        ttk.Button(nav_frame, text="Save & Next", command=self.next_image, 
                  width=12).pack(side=tk.LEFT, padx=5)
        ttk.Button(nav_frame, text="Skip", command=self.skip_image, 
                  width=12).pack(side=tk.LEFT, padx=5)
        
        # Progress indicator
        ttk.Separator(self.ctrl_frame).pack(fill=tk.X, pady=10)
        self.progress_var = tk.StringVar(value="Image 0/0")
        ttk.Label(self.ctrl_frame, textvariable=self.progress_var, 
                 font=("Arial", 10)).pack(pady=5)
        
        # File size indicator
        self.filesize_var = tk.StringVar(value="")
        ttk.Label(self.ctrl_frame, textvariable=self.filesize_var, 
                 font=("Arial", 10)).pack(pady=5)
        
        # Return button
        ttk.Button(self.ctrl_frame, text="Return to Menu", 
                  command=self.show_start_screen, width=20).pack(side=tk.BOTTOM, pady=10)
        
        # Bind keyboard controls for ROI adjustment
        self.bind("<KeyPress>", self.handle_keypress)

    def on_mouse_down(self, event):
        """Handle mouse button press for ROI drawing"""
        if self.img is None:
            return
        
        # Start drawing the rectangle
        self.drawing = True
        
        # Convert to original image coordinates
        self.start_x = int(event.x / self.display_scale)
        self.start_y = int(event.y / self.display_scale)
        
        # Create the initial rectangle on the canvas at the display scale
        canvas_x = event.x
        canvas_y = event.y
        self.rect_id = self.canvas.create_rectangle(
            canvas_x, canvas_y, canvas_x + 1, canvas_y + 1,
            outline='yellow', width=2
        )
        
        self.status_var.set(f"Drawing ROI from ({self.start_x}, {self.start_y})")

    def on_mouse_move(self, event):
        """Handle mouse motion while drawing ROI"""
        if not self.drawing or self.rect_id is None:
            return
        
        # Update rectangle on the canvas
        canvas_x = event.x
        canvas_y = event.y
        start_canvas_x = self.start_x * self.display_scale
        start_canvas_y = self.start_y * self.display_scale
        
        self.canvas.coords(self.rect_id, start_canvas_x, start_canvas_y, canvas_x, canvas_y)
        
        # Calculate the current size in original image coordinates
        current_x = int(event.x / self.display_scale)
        current_y = int(event.y / self.display_scale)
        
        width = abs(current_x - self.start_x)
        height = abs(current_y - self.start_y)
        
        self.status_var.set(f"ROI size: {width}x{height}")

    def on_mouse_up(self, event):
        """Handle mouse button release to finalize ROI"""
        if not self.drawing:
            return
        
        self.drawing = False
        
        # Calculate rectangle in original image coordinates
        end_x = int(event.x / self.display_scale)
        end_y = int(event.y / self.display_scale)
        
        # Calculate center point and dimensions
        cx = (self.start_x + end_x) // 2
        cy = (self.start_y + end_y) // 2
        rw = abs(end_x - self.start_x)
        rh = abs(end_y - self.start_y)
        
        # Ensure valid dimensions
        if rw < 5 or rh < 5:
            self.status_var.set("ROI too small, try again")
            self.canvas.delete(self.rect_id)
            self.rect_id = None
            return
        
        self.roi_rect = [cx, cy, rw, rh]
        
        # Clear the drawing rectangle
        self.canvas.delete(self.rect_id)
        self.rect_id = None
        
        # Redraw with the calculated ROI
        self.display_image()
        self.status_var.set(f"ROI set to center: ({cx}, {cy}), size: {rw}x{rh}")


    def load_image(self):
        if not self.files or self.idx >= len(self.files):
            if self.files:
                messagebox.showinfo("Complete", "All images processed!")
            self.show_start_screen()
            return
        
        # Update progress indicator
        self.progress_var.set(f"Image {self.idx + 1}/{len(self.files)}")
        
        # Load image
        path = self.files[self.idx]
        self.img = cv2.imread(path)
        
        if self.img is None:
            messagebox.showerror("Error", f"Failed to load image: {path}")
            return
        
        # Save original for reference
        self.original_img = self.img.copy()
        
        # Get file size for display
        file_size = os.path.getsize(path)
        size_kb = file_size / 1024
        
        if size_kb < 1000:
            self.filesize_var.set(f"File size: {size_kb:.1f} KB")
        else:
            size_mb = size_kb / 1024
            self.filesize_var.set(f"File size: {size_mb:.2f} MB")
        
        # Extract filename information
        filename = os.path.basename(path)
        self.file_info_var.set(filename)
        
        # Store image dimensions
        self.h, self.w = self.img.shape[:2]
        
        # Reset ROI - user will draw it
        self.roi_rect = None
        
        # Display image
        self.display_image()
        
        # Update window title with image info
        self.title(f"ROI Recropper - {filename}")
        
        # Prompt user to draw ROI
        self.status_var.set("Click and drag to draw a crop rectangle")

    def display_image(self):
        if self.img is None:
            return
        
        # Create a copy for drawing
        display_img = self.img.copy()
        
        # Draw ROI rectangle if one exists
        if self.roi_rect:
            cx, cy, rw, rh = self.roi_rect
            
            # Calculate rectangle corners
            x1 = max(0, cx - rw // 2)
            y1 = max(0, cy - rh // 2)
            x2 = min(self.w, x1 + rw)
            y2 = min(self.h, y1 + rh)
            
            # Draw rectangle
            cv2.rectangle(display_img, (x1, y1), (x2, y2), (0, 255, 255), 2)
            
            # Draw center marker
            cv2.drawMarker(display_img, (cx, cy), (0, 0, 255), cv2.MARKER_CROSS, 10, 2)
        
        # Resize for display
        h, w = display_img.shape[:2]
        self.display_scale = min(PREVIEW_SIZE / w, PREVIEW_SIZE / h)
        display_img = cv2.resize(display_img, (int(w * self.display_scale), int(h * self.display_scale)))
        
        # Convert to PIL format for Tkinter
        display_img = cv2.cvtColor(display_img, cv2.COLOR_BGR2RGB)
        self.photo = ImageTk.PhotoImage(image=Image.fromarray(display_img))
        
        # Display on canvas
        self.canvas.delete("all")
        self.canvas.config(width=self.photo.width(), height=self.photo.height())
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.photo)

    def handle_keypress(self, event):
        if self.roi_rect is None:
            return
            
        cx, cy, rw, rh = self.roi_rect
        
        # Movement with arrow keys
        if event.keysym == 'Up':
            cy -= 1
        elif event.keysym == 'Down':
            cy += 1
        elif event.keysym == 'Left':
            cx -= 1
        elif event.keysym == 'Right':
            cx += 1
        # Size with +/- keys
        elif event.char in ('+', '='):
            rw = min(self.w, rw + 2)
            rh = min(self.h, rh + 2)
        elif event.char == '-':
            rw = max(10, rw - 2)
            rh = max(10, rh - 2)
        # Save and continue with Enter
        elif event.keysym == 'Return':
            self.next_image()
            return
        
        self.roi_rect = [cx, cy, rw, rh]
        self.display_image()

    def save_cropped_roi(self):
        if self.roi_rect is None or self.original_img is None:
            messagebox.showwarning("No ROI", "Please draw a crop region first")
            return False
        
        # Extract ROI using the current rectangle
        cx, cy, rw, rh = self.roi_rect
        
        # Calculate rectangle corners
        x1 = max(0, cx - rw // 2)
        y1 = max(0, cy - rh // 2)
        x2 = min(self.w, x1 + rw)
        y2 = min(self.h, y1 + rh)
        
        # Crop the image
        cropped = self.original_img[y1:y2, x1:x2]
        
        if cropped.size == 0:
            messagebox.showerror("Error", "Cropping region is invalid!")
            return False
        
        # Get current file path
        current_path = self.files[self.idx]
        
        # Preview the crop in a smaller window before saving
        preview_img = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
        preview = Image.fromarray(preview_img)
        
        # Save the cropped image to the same file
        cv2.imwrite(current_path, cropped)
        
        self.status_var.set(f"Saved cropped ROI to {os.path.basename(current_path)}")
        return True

    def next_image(self):
        # Save current crop
        if self.save_cropped_roi():
            # Move to next image
            self.idx += 1
            if self.idx < len(self.files):
                self.load_image()
            else:
                messagebox.showinfo("Complete", "All images processed!")
                self.show_start_screen()

    def prev_image(self):
        # Save current crop
        if self.save_cropped_roi():
            # Move to previous image
            if self.idx > 0:
                self.idx -= 1
                self.load_image()

    def skip_image(self):
        # Skip the current image (move to next without saving)
        self.status_var.set(f"Skipped image: {os.path.basename(self.files[self.idx])}")
        self.idx += 1
        if self.idx < len(self.files):
            self.load_image()
        else:
            messagebox.showinfo("Complete", "All images processed!")
            self.show_start_screen()

if __name__ == "__main__":
    app = ROIRecropperApp()
    app.mainloop()