#!/usr/bin/env python3
"""
analyze_resolution.py

Walks specified folders and files, loads each image, and prints a summary
of:
 - Pixel dimensions
 - Embedded DPI tag
 - Estimated sensor DPI and physical size (mm)
 - File size on disk
 - Mean brightness & contrast
 - Sharpness (variance of Laplacian)

Dependencies:
    pip install pillow opencv-python numpy
"""

import os
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

# Constants for IMX500 sensor
PIXEL_PITCH_UM = 1.55     # micrometers
MM_PER_INCH = 25.4
SENSOR_DPI = MM_PER_INCH / (PIXEL_PITCH_UM * 1e-3)  # ≈ 16387 dpi

# Folders and files to analyze
dir_paths = [
    r"C:\Users\nickh\OneDrive\Documents\nh\Git_repo\coin_sorter3\extracted_pennies\mint_mark",
    r"C:\Users\nickh\OneDrive\Documents\nh\Git_repo\coin_sorter3\extracted_pennies\date",
    r"C:\Users\nickh\OneDrive\Documents\nh\Git_repo\coin_sorter3\extracted_pennies\liberty",
    r"C:\Users\nickh\OneDrive\Documents\nh\Git_repo\coin_sorter3\extracted_pennies\penny",
]
file_paths = [
    r"C:\Users\nickh\OneDrive\Documents\nh\Git_repo\coin_sorter3\50_60_10.png",
    r"C:\Users\nickh\OneDrive\Documents\nh\Git_repo\coin_sorter3\20_30_60.png",
    r"C:\Users\nickh\OneDrive\Documents\nh\Git_repo\coin_sorter3\40_40_50.png",
]

def get_embedded_dpi(path):
    try:
        with Image.open(path) as img:
            info = img.info.get("dpi", None)
            if isinstance(info, tuple) and len(info) >= 2:
                return info
    except Exception:
        pass
    return None

def compute_sharpness(gray):
    # Variance of Laplacian is a standard focus measure
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    return float(lap.var())

def analyze_image(path):
    p = Path(path)
    size_bytes = p.stat().st_size
    # Load via OpenCV
    img = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
    if img is None:
        print(f"[ERROR] Could not open image: {p}")
        return

    h, w = img.shape[:2]
    # Convert to grayscale for brightness/contrast/sharpness
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    mean_brightness = float(np.mean(gray))
    contrast = float(np.std(gray))
    sharpness = compute_sharpness(gray)

    embedded_dpi = get_embedded_dpi(p) or ("N/A", "N/A")

    # Physical size in mm
    width_mm  = w * (PIXEL_PITCH_UM * 1e-3)
    height_mm = h * (PIXEL_PITCH_UM * 1e-3)

    print(f"{p.name}")
    print(f"  Path:         {p}")
    print(f"  File size:    {size_bytes:,} bytes")
    print(f"  Dimensions:   {w}×{h} px")
    print(f"  Embedded DPI: {embedded_dpi[0]}×{embedded_dpi[1]}")
    print(f"  Sensor DPI:   {SENSOR_DPI:.0f} dpi")
    print(f"  Physical:     {width_mm:.1f}×{height_mm:.1f} mm")
    print(f"  Brightness:   μ={mean_brightness:.1f}, σ={contrast:.1f}")
    print(f"  Sharpness:    var(Laplacian) = {sharpness:.1f}")
    print("")

def main():
    print("=== Full-Resolution Analysis ===\n")
    # Process directories
    for d in dir_paths:
        print(f"--- Directory: {d} ---")
        if not os.path.isdir(d):
            print(f"  [WARN] Directory not found, skipping: {d}\n")
            continue
        for fname in sorted(os.listdir(d)):
            full = os.path.join(d, fname)
            if os.path.isfile(full) and fname.lower().endswith((".png", ".jpg", ".jpeg")):
                analyze_image(full)
        print("")

    # Process individual files
    print("--- Individual Files ---")
    for path in file_paths:
        analyze_image(path)

if __name__ == "__main__":
    main()
