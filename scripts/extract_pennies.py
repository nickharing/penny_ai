import cv2
import numpy as np
import os
import tkinter as tk
from tkinter import filedialog
import rawpy

# --- Configuration ---
PREVIEW_SIZE = 1800  # Max dimension for preview window
PADDING_PIXELS = 10  # Padding around the circle when cropping

# --- Helpers ---
def load_image(path):
    """
    Load an image file. Supports DNG via rawpy or standard formats via OpenCV.
    Returns a BGR uint8 image.
    """
    ext = os.path.splitext(path)[1].lower()
    if ext == '.dng':
        with rawpy.imread(path) as raw:
            rgb = raw.postprocess(
                use_camera_wb=True,
                no_auto_bright=True,
                output_bps=8
            )
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    else:
        img = cv2.imread(path)
        if img is None:
            raise FileNotFoundError(f"Cannot open image: {path}")
        return img

class CoinExtractor:
    def __init__(self, image_path, out_dir, max_display=PREVIEW_SIZE):
        # Load full-resolution image
        self.orig = load_image(image_path)
        h, w = self.orig.shape[:2]
        # Compute scale for preview
        self.scale = min(1.0, max_display / max(h, w))
        # Prepare display image
        self.disp = (cv2.resize(self.orig, None, fx=self.scale, fy=self.scale)
                     if self.scale < 1.0 else self.orig.copy())
        self.circle_center = None
        self.circle_radius = int(min(h, w) * 0.18)  # Changed from 0.3 to 0.15 for smaller initial circle
        self.rotation_angle = 0
        self.out_dir = out_dir
        self.movement_step = 1  # Reduced from 5 for finer control
        os.makedirs(out_dir, exist_ok=True)

    def crop_sequence(self, years, uids, coin_side="obverse"):
        """
        Cycle through lists of years and UIDs, allowing the user to click,
        fine-tune a circular crop, then save the result.
        
        Parameters:
        - years: List of years for each penny
        - uids: List of unique identifiers for each penny
        - coin_side: "obverse" or "reverse" to indicate the side being captured
        """
        idx = 0
        win = "Crop Pennies"
        circles = []  # list of (center, radius)
        cv2.namedWindow(win)

        def on_click(evt, x, y, flags, param):
            # Place initial circle center on mouse click
            if evt == cv2.EVENT_LBUTTONDOWN and idx < len(years):
                self.circle_center = (int(x / self.scale), int(y / self.scale))
                self.rotation_angle = 0

        cv2.setMouseCallback(win, on_click)

        while idx < len(years):
            frame = self.disp.copy()
            # Draw previously saved circles
            for c, r in circles:
                cd = (int(c[0] * self.scale), int(c[1] * self.scale))
                rd = int(r * self.scale)
                cv2.circle(frame, cd, rd, (0,255,0), 2)
            # Draw active circle
            if self.circle_center:
                cd = (int(self.circle_center[0] * self.scale),
                      int(self.circle_center[1] * self.scale))
                rd = int(self.circle_radius * self.scale)
                cv2.circle(frame, cd, rd, (0,255,255), 2)
                # Draw padded circle to show the actual crop area
                pd_rd = int((self.circle_radius + PADDING_PIXELS) * self.scale)
                cv2.circle(frame, cd, pd_rd, (0,127,255), 1)
                ang_rad = np.deg2rad(self.rotation_angle - 90)
                end = (int(cd[0] + rd * np.cos(ang_rad)),
                       int(cd[1] + rd * np.sin(ang_rad)))
                cv2.line(frame, cd, end, (0,255,255), 1)
                # Add diameter information
                diameter_text = f"Diameter: {self.circle_radius*2} px (with {PADDING_PIXELS}px padding)"
                cv2.putText(frame, diameter_text, (10,90), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
                move_text = f"Movement step: {self.movement_step}px (use [/] to adjust)"
                cv2.putText(frame, move_text, (10,120), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
            # Status
            text = f"{years[idx]} / UID {uids[idx]}"
            cv2.putText(frame, text, (10,30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255),2)
            cv2.putText(frame, "Click to set circle | arrows: move | +/-: size | 1/2: rotate | [/]: step size | Enter: save | Esc: exit",
                        (10,60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255),1)
            cv2.imshow(win, frame)

            key = cv2.waitKeyEx(1)
            # Exit
            if key == 27:
                break
            # Need click before adjustments
            if not self.circle_center:
                continue
            # Movement
            if key == 2490368: self.circle_center = (self.circle_center[0], self.circle_center[1]-self.movement_step)
            if key == 2621440: self.circle_center = (self.circle_center[0], self.circle_center[1]+self.movement_step)
            if key == 2424832: self.circle_center = (self.circle_center[0]-self.movement_step, self.circle_center[1])
            if key == 2555904: self.circle_center = (self.circle_center[0]+self.movement_step, self.circle_center[1])
            # Adjust movement step size
            if key == ord('['): self.movement_step = max(1, self.movement_step-1)
            if key == ord(']'): self.movement_step = min(10, self.movement_step+1)
            # Resize
            if key in (ord('+'), ord('=')): self.circle_radius += 2
            if key == ord('-'): self.circle_radius = max(5, self.circle_radius-2)
            # Rotate
            if key == ord('1'): self.rotation_angle = (self.rotation_angle + 1) % 360
            if key == ord('2'): self.rotation_angle = (self.rotation_angle - 1) % 360
            # Save crop
            if key == 13:
                H, W = self.orig.shape[:2]
                # Rotate entire image around circle center
                M = cv2.getRotationMatrix2D(self.circle_center, self.rotation_angle, 1)
                rot = cv2.warpAffine(self.orig, M, (W, H), borderValue=(255,255,255))
                
                # Calculate crop area with padding
                padded_radius = self.circle_radius + PADDING_PIXELS
                x1 = max(0, self.circle_center[0] - padded_radius)
                y1 = max(0, self.circle_center[1] - padded_radius)
                x2 = min(W, x1 + 2 * padded_radius)
                y2 = min(H, y1 + 2 * padded_radius)
                crop = rot[int(y1):int(y2), int(x1):int(x2)]
                
                # Format UID with leading zeros to be 4 digits
                uid_str = f"{uids[idx]:04d}"
                # Format the filename according to new convention
                fname = f"penny_{years[idx]}_{coin_side}_S_{uid_str}.jpg"
                path = os.path.join(self.out_dir, fname)
                
                # Check if file already exists and find an alternative name
                base, ext = os.path.splitext(path)
                counter = 1
                while os.path.exists(path):
                    path = f"{base}_{counter}{ext}"
                    counter += 1
                
                cv2.imwrite(path, crop)
                print(f"Saved {os.path.basename(path)}")
                circles.append((self.circle_center, self.circle_radius))
                idx += 1
                self.circle_center = None

        cv2.destroyWindow(win)


def main():
    root = tk.Tk()
    root.withdraw()

    script_dir = os.path.dirname(__file__)
    raw = filedialog.askopenfilename(
        title="Select sheet image",
        initialdir=os.path.join(script_dir, '..', 'raw_images'),
        filetypes=[("Image","*.png;*.jpg;*.jpeg;*.dng")]
    )
    if not raw:
        print("No image selected.")
        return

    out_dir = os.path.join(os.path.dirname(raw), 'extracted_pennies')
    os.makedirs(out_dir, exist_ok=True)

    extractor = CoinExtractor(raw, out_dir)

    coin_side = input("Coin side (obverse/reverse): ").lower()
    if coin_side not in ["obverse", "reverse"]:
        print("Invalid side. Defaulting to 'obverse'.")
        coin_side = "obverse"

    dec = input("Decade (e.g. 50→1950): ")
    if not dec.isdigit():
        print("Invalid decade.")
        return
    d = int(dec)
    year_base = (2000 if d <= 25 else 1900) + d

    uid0 = input("Starting UID: ")
    if not uid0.isdigit():
        print("Invalid UID.")
        return
    uid0 = int(uid0)

    years = [year_base + i for i in range(10)]
    uids = [uid0 + i for i in range(10)]

    extractor.crop_sequence(years, uids, coin_side)


if __name__ == '__main__':
    main()