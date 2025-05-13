import cv2
import numpy as np
import os
import glob
import json
import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
import re

# --- Configuration ---
BASE_DIR = r"C:\Users\nickh\OneDrive\Documents\nh\Git_repo\coin_sorter3"
INPUT_DIR = os.path.join(BASE_DIR, "data", "obverse")
ROI_JSON = os.path.join(BASE_DIR, "image_20250429_092630_data.json")
OUTPUT_DIRS = {
    'date': os.path.join(BASE_DIR, "data", "date"),
    'mint_mark': os.path.join(BASE_DIR, "data", "mint_mark"),
    'liberty': os.path.join(BASE_DIR, "data", "liberty"),
}
PREVIEW_SIZE = 800  # Reduced from 1200 to make the image smaller
ROI_LINE_THICKNESS = 1  # Reduced from 2 to 1 for a thinner line

# Create output directories
for dir_path in OUTPUT_DIRS.values():
    os.makedirs(dir_path, exist_ok=True)

# --- Helper Functions ---
def get_processed_files():
    """
    Scan output directories and return a set of already processed UIDs
    """
    processed_uids = set()
    for roi_type, dir_path in OUTPUT_DIRS.items():
        # Get all image files in the output directory
        output_files = glob.glob(os.path.join(dir_path, f"{roi_type}_*.jpg")) + \
                      glob.glob(os.path.join(dir_path, f"{roi_type}_*.png"))
        
        # Extract UIDs from filenames
        for file_path in output_files:
            filename = os.path.basename(file_path)
            # Match patterns like: liberty_1960_obverse_D_1234.jpg or mint_mark_1960_obverse_D_1234.jpg
            match = re.search(r'_(\d+)\.', filename)
            if match:
                uid = match.group(1)
                processed_uids.add(uid)
    
    print(f"Found {len(processed_uids)} already processed images")
    return processed_uids

# --- ROI Editor App ---
class PennyROIApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Penny ROI Extractor")
        self.geometry("1400x900")
        self.configure(bg='#f0f0f0')
        
        # Load ROI definitions
        with open(ROI_JSON, 'r') as f:
            self.avg_rois = json.load(f)['average_normalized_rois']
        
        # Get already processed files
        self.processed_uids = get_processed_files()
        
        self.create_ui()
        self.mode = None
        self.current_roi = None
        self.files = []
        self.filtered_files = []  # Will store only unprocessed files
        self.idx = 0
        self.img = None
        self.h, self.w = 0, 0
        self.roi_rect = None
        self.need_reload = True
        self.display_scale = 1.0  # Scale factor for display
        
        # Store last used ROI position for each type
        self.last_roi_positions = {
            'date': None,
            'mint_mark': None,
            'liberty': None
        }
        
        # Start screen (mode selection)
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
        
        # Bind canvas click event
        self.canvas.bind("<Button-1>", self.canvas_click)
        
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
        ttk.Label(self.ctrl_frame, text="Penny ROI Extractor", 
                  font=("Arial", 16, "bold")).pack(pady=(0, 20))
        
        # Mode selection buttons
        ttk.Label(self.ctrl_frame, text="Select Task:", 
                  font=("Arial", 12)).pack(pady=(10, 5))
        
        # Large buttons for each task
        button_style = {"width": 20, "padding": 10}
        
        btn_mint = ttk.Button(self.ctrl_frame, text="Edit Mint Marks", 
                            command=lambda: self.start_task('mint_edit'), **button_style)
        btn_mint.pack(pady=5)
        
        btn_date = ttk.Button(self.ctrl_frame, text="Extract Date ROIs", 
                            command=lambda: self.start_task('date'), **button_style)
        btn_date.pack(pady=5)
        
        btn_mint_roi = ttk.Button(self.ctrl_frame, text="Extract Mint Mark ROIs", 
                                command=lambda: self.start_task('mint_mark'), **button_style)
        btn_mint_roi.pack(pady=5)
        
        btn_liberty = ttk.Button(self.ctrl_frame, text="Extract Liberty ROIs", 
                               command=lambda: self.start_task('liberty'), **button_style)
        btn_liberty.pack(pady=5)
        
        # Exit button
        ttk.Button(self.ctrl_frame, text="Exit", 
                 command=self.quit, **button_style).pack(pady=(20, 5))

    def filter_processed_files(self):
        """
        Filter out already processed files based on UID
        """
        filtered = []
        skipped_count = 0
        
        for file_path in self.files:
            filename = os.path.basename(file_path)
            # Extract UID from filename
            uid_match = re.search(r'penny_(\d+)_', filename)
            if not uid_match:
                # Try other pattern
                uid_match = re.search(r'_(\d+)\.', filename)
            
            if uid_match:
                uid = uid_match.group(1)
                # Only include files that haven't been processed
                if uid not in self.processed_uids:
                    filtered.append(file_path)
                else:
                    skipped_count += 1
            else:
                # Include files with unrecognized pattern (to be safe)
                filtered.append(file_path)
        
        return filtered, skipped_count

    def start_task(self, task_mode):
        self.mode = task_mode
        
        if task_mode == 'mint_edit':
            # For mint mark editing, find files and show mint editing UI
            self.files = sorted(glob.glob(os.path.join(INPUT_DIR, 'penny_*.jpg')))
            # Don't filter for mint_edit mode
            self.filtered_files = self.files
            self.show_mint_editor()
        else:
            # For ROI extraction tasks
            self.current_roi = task_mode
            self.files = sorted(glob.glob(os.path.join(INPUT_DIR, 'penny_*.jpg')))
            
            # Filter out already processed files
            self.filtered_files, skipped_count = self.filter_processed_files()
            
            # Show information about skipped files
            if skipped_count > 0:
                messagebox.showinfo(
                    "Skipping Processed Files", 
                    f"{skipped_count} images have already been processed and will be skipped.\n"
                    f"{len(self.filtered_files)} images remaining to process."
                )
            
            self.show_roi_editor()
        
        self.idx = 0
        if self.filtered_files:
            self.load_image()
        else:
            messagebox.showinfo("No Files", "No unprocessed files found!")
            self.show_start_screen()

    def show_mint_editor(self):
        # Clear existing controls
        for widget in self.ctrl_frame.winfo_children():
            widget.destroy()
        
        # Title
        ttk.Label(self.ctrl_frame, text="Edit Mint Mark", 
                  font=("Arial", 14, "bold")).pack(pady=(0, 20))
        
        # Mint mark selection
        mint_frame = ttk.Frame(self.ctrl_frame)
        mint_frame.pack(pady=10, fill=tk.X)
        
        ttk.Label(mint_frame, text="Mint Mark:", font=("Arial", 12)).pack(pady=(0, 5))
        
        # Mint mark option buttons (large and easy to click)
        self.mint_var = tk.StringVar(value='nomint')
        
        mint_buttons_frame = ttk.Frame(mint_frame)
        mint_buttons_frame.pack(fill=tk.X)
        
        button_style = {"width": 8, "padding": 10}
        
        ttk.Radiobutton(mint_buttons_frame, text="S", variable=self.mint_var, 
                       value='S', **button_style).pack(side=tk.LEFT, padx=2)
        ttk.Radiobutton(mint_buttons_frame, text="D", variable=self.mint_var, 
                       value='D', **button_style).pack(side=tk.LEFT, padx=2)
        ttk.Radiobutton(mint_buttons_frame, text="P", variable=self.mint_var, 
                       value='P', **button_style).pack(side=tk.LEFT, padx=2)
        ttk.Radiobutton(mint_buttons_frame, text="W", variable=self.mint_var, 
                       value='W', **button_style).pack(side=tk.LEFT, padx=2)
        
        ttk.Radiobutton(mint_frame, text="No Mint Mark", variable=self.mint_var, 
                       value='nomint', **button_style).pack(pady=(5, 0))
        
        # Navigation buttons
        nav_frame = ttk.Frame(self.ctrl_frame)
        nav_frame.pack(pady=20, fill=tk.X)
        
        ttk.Button(nav_frame, text="Save & Prev", command=self.prev_image, 
                  width=12).pack(side=tk.LEFT, padx=5)
        ttk.Button(nav_frame, text="Save & Next", command=self.next_image, 
                  width=12).pack(side=tk.LEFT, padx=5)
        ttk.Button(nav_frame, text="Skip", command=self.skip_image, 
                  width=12).pack(side=tk.LEFT, padx=5)
        ttk.Button(nav_frame, text="Delete", command=self.delete_image, 
                  width=12).pack(side=tk.LEFT, padx=5)
        
        # Progress indicator
        ttk.Separator(self.ctrl_frame).pack(fill=tk.X, pady=10)
        self.progress_var = tk.StringVar(value="Image 0/0")
        ttk.Label(self.ctrl_frame, textvariable=self.progress_var, 
                 font=("Arial", 10)).pack(pady=5)
        
        # Return button
        ttk.Button(self.ctrl_frame, text="Return to Menu", 
                  command=self.show_start_screen, width=20).pack(side=tk.BOTTOM, pady=10)

    def show_roi_editor(self):
        # Clear existing controls
        for widget in self.ctrl_frame.winfo_children():
            widget.destroy()
        
        # Title based on current ROI
        roi_titles = {
            'date': "Extract Date ROI",
            'mint_mark': "Extract Mint Mark ROI",
            'liberty': "Extract Liberty ROI"
        }
        
        ttk.Label(self.ctrl_frame, text=roi_titles.get(self.current_roi, "Extract ROI"), 
                  font=("Arial", 14, "bold")).pack(pady=(0, 20))
        
        # ROI adjustment instructions
        instruction_text = """
        Controls:
        
        Click: Move ROI to position
        Arrow keys: Fine-tune position
        +/- keys: Resize ROI
        1/2 keys: Rotate ROI
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
        ttk.Button(nav_frame, text="Delete", command=self.delete_image, 
                  width=12).pack(side=tk.LEFT, padx=5)
        
        # Progress indicator
        ttk.Separator(self.ctrl_frame).pack(fill=tk.X, pady=10)
        self.progress_var = tk.StringVar(value="Image 0/0")
        ttk.Label(self.ctrl_frame, textvariable=self.progress_var, 
                 font=("Arial", 10)).pack(pady=5)
        
        # Return button
        ttk.Button(self.ctrl_frame, text="Return to Menu", 
                  command=self.show_start_screen, width=20).pack(side=tk.BOTTOM, pady=10)
        
        # Bind keyboard controls for ROI adjustment
        self.bind("<KeyPress>", self.handle_keypress)

    def canvas_click(self, event):
        """Handle click on canvas to position ROI"""
        if self.mode != 'mint_edit' and self.roi_rect is not None:
            # Convert click coordinates from display scale to original image scale
            orig_x = int(event.x / self.display_scale)
            orig_y = int(event.y / self.display_scale)
            
            # Update ROI center position, keeping other parameters the same
            _, _, rw, rh, ang = self.roi_rect
            self.roi_rect = [orig_x, orig_y, rw, rh, ang]
            self.display_image()
            self.status_var.set(f"ROI moved to ({orig_x}, {orig_y})")

    def load_image(self):
        if not self.filtered_files or self.idx >= len(self.filtered_files):
            if self.filtered_files:
                messagebox.showinfo("Complete", "All images processed!")
            self.show_start_screen()
            return
        
        # Update progress indicator
        self.progress_var.set(f"Image {self.idx + 1}/{len(self.filtered_files)}")
        
        # Load image
        path = self.filtered_files[self.idx]
        self.img = cv2.imread(path)
        if self.img is None:
            messagebox.showerror("Error", f"Failed to load image: {path}")
            return
        
        self.h, self.w = self.img.shape[:2]
        
        # Extract filename components
        filename = os.path.basename(path)
        match = re.search(r'penny_(\d+)_([^_]+)_([^_]+)_(\d+)', filename)
        if match:
            self.year, self.side, self.mint, self.uid = match.groups()
        else:
            # Try older naming pattern
            match = re.search(r'penny_(\d+)_(\d+)_M', filename)
            if match:
                self.year, self.uid = match.groups()
                self.side = "obverse"  # Default
                self.mint = "nomint"   # Default
        
        # Initialize ROI if in ROI edit mode
        if self.mode != 'mint_edit':
            # Check if we have a previously used position for this ROI type
            if self.last_roi_positions[self.current_roi] is not None:
                # Use previous position but scaled to this image size
                prev_roi = self.last_roi_positions[self.current_roi]
                # Extract normalized coordinates from previous ROI
                prev_cx_norm = prev_roi[0] / prev_roi[4]  # cx / prev_width
                prev_cy_norm = prev_roi[1] / prev_roi[5]  # cy / prev_height
                prev_rw_norm = prev_roi[2] / prev_roi[4]  # rw / prev_width
                prev_rh_norm = prev_roi[3] / prev_roi[5]  # rh / prev_height
                
                # Apply to current image
                cx = int(prev_cx_norm * self.w)
                cy = int(prev_cy_norm * self.h)
                rw = int(prev_rw_norm * self.w)
                rh = int(prev_rh_norm * self.h)
                ang = prev_roi[6]  # Keep the same angle
                
                self.roi_rect = [cx, cy, rw, rh, ang]
            else:
                # Initialize from average normalized coordinates
                norm_roi = self.avg_rois.get(self.current_roi, [0.5, 0.5, 0.2, 0.1])
                cx = int(norm_roi[0] * self.w)
                cy = int(norm_roi[1] * self.h)
                rw = int(norm_roi[2] * self.w)
                rh = int(norm_roi[3] * self.h)
                self.roi_rect = [cx, cy, rw, rh, 0]  # [cx, cy, width, height, angle]
        
        # Set mint mark variable if in mint editing mode
        if self.mode == 'mint_edit':
            self.mint_var.set(self.mint if self.mint else 'nomint')
        
        # Display image
        self.display_image()
        
        # Update window title with image info
        self.title(f"Penny ROI Extractor - {self.year} - {self.side}")

    def display_image(self):
        if self.img is None:
            return
        
        # Create a copy for drawing
        display_img = self.img.copy()
        
        # Draw ROI if in ROI editing mode
        if self.mode != 'mint_edit' and self.roi_rect:
            cx, cy, rw, rh, ang = self.roi_rect
            rect = ((cx, cy), (rw, rh), ang)
            box = cv2.boxPoints(rect).astype(int)
            cv2.drawContours(display_img, [box], 0, (0, 255, 255), ROI_LINE_THICKNESS)  # Thinner line
            
            # Draw centerline for orientation reference
            rad = np.deg2rad(ang - 90)
            dx, dy = int(rw/2 * np.cos(rad)), int(rw/2 * np.sin(rad))
            cv2.line(display_img, (cx, cy), (cx + dx, cy + dy), (0, 0, 255), ROI_LINE_THICKNESS)
        
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
        if self.mode == 'mint_edit' or self.roi_rect is None:
            return
            
        cx, cy, rw, rh, ang = self.roi_rect
        
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
            rw += 1
            rh += 1
        elif event.char == '-':
            rw = max(5, rw - 1)
            rh = max(5, rh - 1)
        # Rotation with 1/2 keys
        elif event.char == '1':
            ang = (ang + 1) % 360
        elif event.char == '2':
            ang = (ang - 1) % 360
        # Save and continue with Enter
        elif event.keysym == 'Return':
            self.next_image()
            return
        
        self.roi_rect = [cx, cy, rw, rh, ang]
        self.display_image()

    def extract_and_save_roi(self):
        if self.mode == 'mint_edit':
            # Update mint mark in filename
            new_mint = self.mint_var.get()
            old_path = self.filtered_files[self.idx]
            filename = os.path.basename(old_path)
            
            # Use regex to create new filename with updated mint mark
            if '_obverse_' in filename or '_reverse_' in filename:
                parts = re.split(r'_', filename)
                if len(parts) >= 5:  # penny_year_obverse_S_uid.png
                    parts[3] = new_mint
                    new_filename = '_'.join(parts)
                    new_path = os.path.join(os.path.dirname(old_path), new_filename)
                    
                    # Rename the file
                    os.rename(old_path, new_path)
                    self.filtered_files[self.idx] = new_path
                    self.status_var.set(f"Updated mint mark to {new_mint}")
            else:
                # Handle older naming convention if needed
                messagebox.showwarning("Warning", "Filename format not recognized for mint update")
        elif self.current_roi and self.roi_rect:
            # Extract ROI using the current rectangle
            cx, cy, rw, rh, ang = self.roi_rect
            
            # Store current position for next image, including image dimensions
            self.last_roi_positions[self.current_roi] = [cx, cy, rw, rh, self.w, self.h, ang]
            
            # Apply rotation
            M = cv2.getRotationMatrix2D((cx, cy), ang, 1)
            rotated = cv2.warpAffine(self.img, M, (self.w, self.h), borderValue=(255, 255, 255))
            
            # Extract rectangle region
            x1 = max(0, cx - rw // 2)
            y1 = max(0, cy - rh // 2)
            x2 = min(self.w, x1 + rw)
            y2 = min(self.h, y1 + rh)
            roi = rotated[y1:y2, x1:x2]
            
            # Create output filename
            base_filename = os.path.basename(self.filtered_files[self.idx])
            parts = re.split(r'_', base_filename)
            
            # Check file format
            if len(parts) >= 5:  # penny_year_obverse_S_uid.png
                year = parts[1]
                side = parts[2]
                mint = parts[3]
                uid = parts[4].split('.')[0]
                ext = os.path.splitext(base_filename)[1]
                
                # Create new filename: roi-type_year_side_mint_uid.ext
                out_filename = f"{self.current_roi}_{year}_{side}_{mint}_{uid}{ext}"
                out_path = os.path.join(OUTPUT_DIRS[self.current_roi], out_filename)
                
                # Save the ROI
                cv2.imwrite(out_path, roi)
                
                # Add this UID to the processed set
                self.processed_uids.add(uid)
                
                self.status_var.set(f"Saved ROI to {out_filename}")
            else:
                messagebox.showwarning("Warning", "Filename format not recognized for ROI extraction")

    def next_image(self):
        # Save current changes
        self.extract_and_save_roi()
        
        # Move to next image
        self.idx += 1
        if self.idx < len(self.filtered_files):
            self.load_image()
        else:
            messagebox.showinfo("Complete", "All images processed!")
            self.show_start_screen()

    def prev_image(self):
        # Save current changes
        self.extract_and_save_roi()
        
        # Move to previous image
        if self.idx > 0:
            self.idx -= 1
            self.load_image()

    def skip_image(self):
        # Skip the current image (move to next without saving)
        self.status_var.set(f"Skipped image: {os.path.basename(self.filtered_files[self.idx])}")
        self.idx += 1
        if self.idx < len(self.filtered_files):
            self.load_image()
        else:
            messagebox.showinfo("Complete", "All images processed!")
            self.show_start_screen()

    def delete_image(self):
        # Option to flag or delete the current image
        if messagebox.askyesno("Confirm Delete", "Are you sure you want to delete this image?"):
            path = self.filtered_files[self.idx]
            try:
                # Move to trash or delete
                # os.remove(path)  # Uncomment to actually delete
                # For now, just move to next
                self.status_var.set(f"Flagged image for deletion: {os.path.basename(path)}")
                self.next_image()
            except Exception as e:
                messagebox.showerror("Error", f"Failed to delete image: {str(e)}")

if __name__ == "__main__":
    app = PennyROIApp()
    app.mainloop()