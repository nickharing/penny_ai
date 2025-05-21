"""
Filename: detect_penny.py
Purpose:  Detects and crops a penny from input images.
          Can operate on a single image or batch process a directory.
Author:   Penny Vision Team (Originally Gemini Code Assist)
Date:     2025-05-03
Version:  0.4.0

Dependencies:
  - opencv-python
  - numpy
  - PyYAML

Usage:
  Single image (full path):
    python src/detect_penny.py path/to/your/penny.jpg -o path/to/output/cropped_penny.png

  Single image (filename from config's raw_images dir):
    python src/detect_penny.py --filename penny_image.jpg

  Batch process (all images in config's data_root/obverse and data_root/reverse dirs):
    python src/detect_penny.py
"""

# Standard library imports
import argparse
import logging
from pathlib import Path

# Third-party imports
import cv2
import numpy as np
import yaml

# ---***---*** Project Root Definition ***---***---
# Assuming this script is in 'src/', so project root is two levels up.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
# ---***---*** End Project Root Definition ***---***---

# ---***---*** Logging Setup ***---***---
logger = logging.getLogger(__name__)
# ---***---*** End Logging Setup ***---***---

# ---***---*** Config Path ***---***---
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "training_config.yaml"
# ---***---*** End Config Path ***---***---

# ---***---*** Constants ***---***---
DEFAULT_HOUGH_DP = 1.2
DEFAULT_HOUGH_MINDIST_DIVISOR = 4
DEFAULT_HOUGH_PARAM1 = 100
DEFAULT_HOUGH_PARAM2 = 30
DEFAULT_HOUGH_MINRADIUS_DIVISOR = 12
DEFAULT_HOUGH_MAXRADIUS_DIVISOR = 2
BATCH_OUTPUT_SUBDIR_NAME = "detected_pennies_batch" # Subdirectory for batch processed images
# ---***---*** End Constants ***---***---


def load_config(config_path: Path) -> dict:
    """
    Loads the YAML configuration file.

    Args:
        config_path (Path): The path to the YAML configuration file.

    Returns:
        dict: The loaded configuration as a dictionary. Returns an empty
              dictionary if the file is not found or an error occurs.
    """
    if not config_path.exists():
        logger.warning(
            f"Configuration file not found at {config_path}. Using default behaviors."
        )
        return {}
    try:
        with open(config_path, 'r') as f:
            config_data = yaml.safe_load(f)
        logger.info(f"Successfully loaded configuration from {config_path}")
        return config_data
    except Exception as e:
        logger.error(f"Error loading configuration from {config_path}: {e}", exc_info=True)
        return {}


def get_single_img_io_paths_from_config(
    config: dict, input_image_filename: str
) -> tuple[Path | None, Path | None]:
    """
    Determines input and output paths for a single image using config.
    Used when --filename argument is provided.

    Args:
        config (dict): The loaded configuration dictionary.
        input_image_filename (str): The filename of the input image.

    Returns:
        tuple[Path | None, Path | None]: A tuple containing the input Path (or None)
                                         and the default output directory Path for single images.
    """
    paths_config = config.get("paths", {})
    base_data_root_str = paths_config.get("data_root", "./data")
    base_data_root = PROJECT_ROOT / base_data_root_str
    
    # Assuming raw images for single file processing are in a 'raw_images' subdir
    # This convention should be documented or made configurable if it varies.
    default_raw_dir = base_data_root / "raw_images"
    
    # Default output for single processed images, can be overridden by -o
    # Using 'exports' as a general output area, or a specific 'detected_pennies' under it
    default_output_dir_str = paths_config.get("exports", "./output/exports")
    default_output_dir = PROJECT_ROOT / default_output_dir_str / "detected_pennies_single"

    input_path = default_raw_dir / input_image_filename
    return input_path, default_output_dir


def detect_and_crop_penny(
    image_path: Path,
    output_path: Path | None = None,
    dp: float = DEFAULT_HOUGH_DP,
    mindist_divisor: int = DEFAULT_HOUGH_MINDIST_DIVISOR,
    param1: int = DEFAULT_HOUGH_PARAM1,
    param2: int = DEFAULT_HOUGH_PARAM2,
    minradius_divisor: int = DEFAULT_HOUGH_MINRADIUS_DIVISOR,
    maxradius_divisor: int = DEFAULT_HOUGH_MAXRADIUS_DIVISOR,
) -> np.ndarray | None:
    """
    Detects a penny in an image using OpenCV's Hough Circle Transform
    and crops the image to the detected penny.

    Args:
        image_path (Path): Path to the input image file.
        output_path (Path, optional): Path to save the cropped penny image.
                                      If None, the image is not saved. Defaults to None.
        dp (float): HoughCircles dp parameter.
        mindist_divisor (int): Divisor for image's min dimension to set minDist.
        param1 (int): HoughCircles param1 (Canny upper threshold).
        param2 (int): HoughCircles param2 (accumulator threshold).
        minradius_divisor (int): Divisor for image's min dimension to set minRadius.
        maxradius_divisor (int): Divisor for image's min dimension to set maxRadius.

    Returns:
        np.ndarray | None: The cropped penny image as a NumPy array if a penny is detected,
                           otherwise None.
    Raises:
        FileNotFoundError: If the image_path does not exist.
    """
    if not image_path.exists():
        logger.error(f"Image file not found: {image_path}")
        raise FileNotFoundError(f"Image file not found: {image_path}")

    try:
        image = cv2.imread(str(image_path))
        if image is None:
            logger.error(f"Failed to load image: {image_path}")
            return None
        # logger.info(f"Successfully loaded image: {image_path} with shape {image.shape}") # Can be too verbose for batch

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (9, 9), 2)
        # logger.debug("Converted image to grayscale and applied Gaussian blur.") # Verbose for batch

        h, w = blurred.shape
        min_dimension = min(h, w)

        min_dist_val = max(1, min_dimension // mindist_divisor)
        min_radius_val = max(1, min_dimension // minradius_divisor)
        max_radius_val = max(1, min_dimension // maxradius_divisor)

        # logger.debug(
        #     f"HoughCircles params for {image_path.name}: dp={dp}, minDist={min_dist_val}, "
        #     f"param1={param1}, param2={param2}, "
        #     f"minRadius={min_radius_val}, maxRadius={max_radius_val}"
        # ) # Verbose for batch

        circles = cv2.HoughCircles(
            blurred,
            cv2.HOUGH_GRADIENT,
            dp=dp,
            minDist=min_dist_val,
            param1=param1,
            param2=param2,
            minRadius=min_radius_val,
            maxRadius=max_radius_val,
        )

        if circles is not None:
            circles_uint16 = np.uint16(np.around(circles))
            best_circle = None

            if len(circles_uint16[0, :]) > 0:
                sorted_circles = sorted(circles_uint16[0, :], key=lambda c: c[2], reverse=True)
                best_circle = sorted_circles[0]
                # logger.info(
                #     f"Detected {len(circles_uint16[0, :])} circle(s) in {image_path.name}. "
                #     f"Selected largest: (x={best_circle[0]}, y={best_circle[1]}), r={best_circle[2]}"
                # ) # Verbose for batch

            if best_circle is not None:
                x, y, r = best_circle
                x1 = max(0, x - r)
                y1 = max(0, y - r)
                x2 = min(image.shape[1], x + r)
                y2 = min(image.shape[0], y + r)

                cropped_penny = image[y1:y2, x1:x2]

                if cropped_penny.size == 0:
                    logger.warning(f"Cropped penny is empty for {image_path.name}. Check coordinates and radius.")
                    return None

                # logger.info(f"Cropped {image_path.name} to bounding box: [x1={x1}, y1={y1}, x2={x2}, y2={y2}]") # Verbose

                if output_path:
                    try:
                        output_path.parent.mkdir(parents=True, exist_ok=True)
                        cv2.imwrite(str(output_path), cropped_penny)
                        # logger.info(f"Cropped penny from {image_path.name} saved to: {output_path}") # Verbose
                    except Exception as e_save:
                        logger.error(f"Failed to save cropped image from {image_path.name} to {output_path}: {e_save}")
                return cropped_penny
            else:
                logger.warning(f"No suitable circle found after filtering for {image_path.name}.")
                return None
        else:
            logger.warning(f"No circles detected in {image_path.name}.")
            return None

    except FileNotFoundError:
        raise
    except Exception as e_proc:
        logger.error(f"An error occurred during penny detection/cropping for {image_path.name}: {e_proc}", exc_info=True)
        return None # Return None on error to allow batch processing to continue


def main():
    """
    Main function to parse arguments, load configuration, and run penny detection
    in single image or batch mode.
    """
    parser = argparse.ArgumentParser(
        description="Detect and crop a penny from an image or batch of images.",
        formatter_class=argparse.RawTextHelpFormatter # For better help text formatting
    )
    parser.add_argument(
        "input_image", nargs="?", type=Path, default=None,
        help=(
            "Optional. Path to a specific input image for single image processing.\n"
            "If omitted, the script attempts batch processing based on config."
        )
    )
    parser.add_argument(
        "-o", "--output_image", type=Path, default=None,
        help=(
            "Optional. Path to save the cropped penny image in single image mode.\n"
            "Overrides config-based output naming if 'input_image' or '--filename' is specified.\n"
            "Not used in batch mode (output directory is determined by config)."
        )
    )
    parser.add_argument(
        "--config", type=Path, default=DEFAULT_CONFIG_PATH,
        help=f"Path to the training_config.yaml file. Defaults to:\n{DEFAULT_CONFIG_PATH}"
    )
    parser.add_argument(
        "-f", "--filename", type=str, default=None,
        help=(
            "Optional. Filename of an image (from config's raw_images dir) for single image processing.\n"
            "Use if 'input_image' path is not fully specified. Ignored in batch mode."
        )
    )
    # HoughCircles parameters
    hough_group = parser.add_argument_group('HoughCircles Parameters (for detection tuning)')
    hough_group.add_argument("--dp", type=float, default=DEFAULT_HOUGH_DP, help="HoughCircles: dp (default: %(default)s)")
    hough_group.add_argument("--mindist_divisor", type=int, default=DEFAULT_HOUGH_MINDIST_DIVISOR, help="HoughCircles: minDist divisor (default: %(default)s)")
    hough_group.add_argument("--param1", type=int, default=DEFAULT_HOUGH_PARAM1, help="HoughCircles: param1 (Canny upper threshold) (default: %(default)s)")
    hough_group.add_argument("--param2", type=int, default=DEFAULT_HOUGH_PARAM2, help="HoughCircles: param2 (accumulator threshold) (default: %(default)s)")
    hough_group.add_argument("--minradius_divisor", type=int, default=DEFAULT_HOUGH_MINRADIUS_DIVISOR, help="HoughCircles: minRadius divisor (default: %(default)s)")
    hough_group.add_argument("--maxradius_divisor", type=int, default=DEFAULT_HOUGH_MAXRADIUS_DIVISOR, help="HoughCircles: maxRadius divisor (default: %(default)s)")

    parser.add_argument(
        "-v", "--verbose", action="count", default=0,
        help="Increase logging verbosity (-v for INFO, -vv for DEBUG)."
    )

    args = parser.parse_args()

    # Setup logging level
    if args.verbose == 1:
        log_level = logging.INFO
    elif args.verbose >= 2:
        log_level = logging.DEBUG
    else:
        log_level = logging.WARNING # Default to WARNING if no -v

    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler()],
    )

    config = load_config(args.config)
    if not config:
        logger.error("Failed to load configuration. Exiting.")
        return

    paths_config = config.get("paths", {})
    base_data_root_str = paths_config.get("data_root", "./data") # Relative to project root
    base_data_root_abs = (PROJECT_ROOT / base_data_root_str).resolve()


    # --- Mode Determination ---
    if args.input_image or args.filename:
        # --- Single Image Mode ---
        logger.info("Operating in SINGLE IMAGE mode.")
        final_input_path = None
        final_output_path = None # For the cropped image

        if args.input_image:
            final_input_path = args.input_image.resolve()
            # If output_image is not given, construct it based on input filename and config output dir
            _, default_single_output_dir = get_single_img_io_paths_from_config(config, final_input_path.name) # Need a filename for this helper
            if args.output_image:
                final_output_path = args.output_image.resolve()
            else:
                default_single_output_dir.mkdir(parents=True, exist_ok=True)
                final_output_path = default_single_output_dir / f"cropped_{final_input_path.name}"
        elif args.filename: # args.input_image is None, but args.filename is provided
            config_input_path_for_file, default_single_output_dir = get_single_img_io_paths_from_config(config, args.filename)
            if config_input_path_for_file and config_input_path_for_file.exists():
                final_input_path = config_input_path_for_file.resolve()
            else:
                logger.error(f"Input file '{args.filename}' not found in configured raw images directory: {config_input_path_for_file}")
                return

            if args.output_image:
                final_output_path = args.output_image.resolve()
            else:
                default_single_output_dir.mkdir(parents=True, exist_ok=True)
                final_output_path = default_single_output_dir / f"cropped_{args.filename}"
        
        if final_input_path:
            logger.info(f"Processing single image: {final_input_path}")
            cropped_image = detect_and_crop_penny(
                final_input_path, final_output_path,
                dp=args.dp, mindist_divisor=args.mindist_divisor,
                param1=args.param1, param2=args.param2,
                minradius_divisor=args.minradius_divisor, maxradius_divisor=args.maxradius_divisor
            )
            if cropped_image is not None:
                logger.info(f"Successfully processed and saved to {final_output_path if final_output_path else 'memory (no output path)'}.")
            else:
                logger.warning(f"Failed to process {final_input_path}.")
        else:
            logger.error("No valid input path determined for single image mode.")

    else:
        # --- Batch Processing Mode ---
        logger.info("Operating in BATCH PROCESSING mode (no specific input_image or filename provided).")
        
        # Define subdirectories under data_root to search for images
        source_subdirs = ["obverse", "reverse"] # Target directories as per user input
        
        # Define base output directory for all batch processed images
        # Convention: <data_root>/<BATCH_OUTPUT_SUBDIR_NAME>/
        base_batch_output_dir = base_data_root_abs / BATCH_OUTPUT_SUBDIR_NAME
        base_batch_output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Base batch output directory for cropped images: {base_batch_output_dir}")

        total_images_found = 0
        total_success_count = 0
        total_failure_count = 0

        for subdir_name in source_subdirs:
            current_batch_input_dir = base_data_root_abs / subdir_name
            current_batch_output_dir = base_batch_output_dir / subdir_name

            if not current_batch_input_dir.exists() or not current_batch_input_dir.is_dir():
                logger.warning(f"Source subdirectory for batch processing not found or not a directory: {current_batch_input_dir}")
                continue # Skip to the next source subdirectory

            current_batch_output_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Processing images from: {current_batch_input_dir}")
            logger.info(f"Saving cropped images to: {current_batch_output_dir}")

            image_files = list(current_batch_input_dir.glob('*.[jJ][pP][gG]')) + \
                          list(current_batch_input_dir.glob('*.[jJ][pP][eE][gG]')) + \
                          list(current_batch_input_dir.glob('*.[pP][nN][gG]'))

            if not image_files:
                logger.info(f"No image files (.jpg, .jpeg, .png) found in {current_batch_input_dir}")
                continue

            logger.info(f"Found {len(image_files)} images in {current_batch_input_dir} to process.")
            total_images_found += len(image_files)

            for image_file_path in image_files:
                logger.debug(f"Processing: {image_file_path.name} from {subdir_name}")
                output_file_path = current_batch_output_dir / f"cropped_{image_file_path.name}"
                
                try:
                    cropped_image = detect_and_crop_penny(
                        image_file_path, output_file_path,
                        dp=args.dp, mindist_divisor=args.mindist_divisor,
                        param1=args.param1, param2=args.param2,
                        minradius_divisor=args.minradius_divisor, maxradius_divisor=args.maxradius_divisor
                    )
                    if cropped_image is not None:
                        total_success_count += 1
                        logger.debug(f"Successfully processed and saved: {output_file_path.name}")
                    else:
                        total_failure_count += 1
                        logger.warning(f"Failed to detect/crop penny in: {image_file_path.name}")
                except Exception as e_batch_item:
                    total_failure_count += 1
                    logger.error(f"Error processing {image_file_path.name} in batch: {e_batch_item}", exc_info=log_level <= logging.DEBUG)


        logger.info("--- Batch Processing Summary ---")
        logger.info(f"Total images found across specified subdirectories: {total_images_found}")
        logger.info(f"Successfully cropped: {total_success_count}")
        logger.info(f"Failed to crop: {total_failure_count}")
        logger.info(f"Cropped images saved under: {base_batch_output_dir}")


if __name__ == "__main__":
    main()
