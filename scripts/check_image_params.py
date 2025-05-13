# check_image_params.py
# Reads an image file and prints its basic parameters.

import cv2
import os

# --- Configuration ---
# Use the same hardcoded path as in your main script
image_path = r"C:\Users\nickh\OneDrive\Documents\nh\Git_repo\coin_sorter3\40_40_50.png"
# --- End Configuration ---

# Normalize the path (good practice for cross-platform compatibility)
normalized_path = os.path.normpath(image_path)

# Check if the file exists
if not os.path.exists(normalized_path):
    print(f"Error: Image file not found at '{normalized_path}'")
else:
    # Attempt to read the image
    try:
        # Read the image using OpenCV
        # cv2.IMREAD_UNCHANGED allows reading alpha channel if present
        img = cv2.imread(normalized_path, cv2.IMREAD_UNCHANGED)

        if img is None:
            print(f"Error: Could not read the image file. It might be corrupted or in an unsupported format.")
        else:
            print(f"--- Parameters for: {normalized_path} ---")

            # Get dimensions (height, width, channels)
            if img.ndim == 2:
                # Grayscale image
                height, width = img.shape
                channels = 1
                print(f"Resolution: {width} x {height} pixels")
                print(f"Color Space: Grayscale ({channels} channel)")
            elif img.ndim == 3:
                # Color image (or color + alpha)
                height, width, channels = img.shape
                print(f"Resolution: {width} x {height} pixels")
                if channels == 3:
                    print(f"Color Space: BGR ({channels} channels)") # OpenCV reads as BGR by default
                elif channels == 4:
                    print(f"Color Space: BGRA ({channels} channels - includes alpha)")
                else:
                    print(f"Color Space: Unknown ({channels} channels)")
            else:
                print(f"Image dimensions unusual: {img.shape}")

            # Get data type (bit depth)
            # Common types: uint8 (8-bit), uint16 (16-bit), float32 (32-bit float)
            print(f"Data Type: {img.dtype}")

    except Exception as e:
        print(f"An error occurred while trying to read the image: {e}")

