# generate_directory_report.py
import argparse
from pathlib import Path
from collections import Counter
import random # Added for sampling files

# Common directories and file patterns to ignore during the detailed tree walk
# and in the summary counts for clarity.
IGNORED_DIRS = {
    "__pycache__",
    ".git",
    ".pytest_cache",
    ".venv",
    "venv",
    "env",
    ".env",
    "AI_env_311", # Specific to your environment, can be generalized
    "node_modules",
    ".vscode",
    ".idea",
    "build",
    "dist",
    "*.egg-info",
}

# File extensions to specifically count in the summary
COUNTED_EXTENSIONS = {
    ".py": "Python Files",
    ".ipynb": "Jupyter Notebooks",
    ".json": "JSON Files",
    ".yaml": "YAML Files",
    ".yml": "YAML Files",
    ".md": "Markdown Files",
    ".txt": "Text Files",
    ".csv": "CSV Files",
    ".pt": "PyTorch Model Files",
    ".onnx": "ONNX Model Files",
    ".jpg": "JPEG Images",
    ".jpeg": "JPEG Images",
    ".png": "PNG Images",
    ".gif": "GIF Images",
    ".bmp": "BMP Images",
    ".log": "Log Files",
    ".sh": "Shell Scripts",
    ".bat": "Batch Scripts",
}


def print_tree(directory: Path, prefix: str = "", report_lines: list = None, level: int = 0):
    """
    Recursively prints a directory tree structure.
    """
    if report_lines is None:
        report_lines = []
    max_files_to_list_in_image_folder = 10

    # Filter out ignored directories at the current level
    contents = [item for item in directory.iterdir() if item.name not in IGNORED_DIRS and not any(item.match(pattern) for pattern in IGNORED_DIRS if "*" in pattern)]
    
    # Separate directories and files for sorted printing
    directories = sorted([item for item in contents if item.is_dir()], key=lambda p: p.name.lower())
    files = sorted([item for item in contents if item.is_file()], key=lambda p: p.name.lower())

    # Process directories first
    for i, path in enumerate(directories):
        connector = "├── " if i < len(directories) - 1 or files else "└── "
        report_lines.append(f"{prefix}{connector}[D] {path.name}")
        if i < len(directories) - 1 or files:
            new_prefix = prefix + "│   "
        else:
            new_prefix = prefix + "    "
        print_tree(path, prefix=new_prefix, report_lines=report_lines, level=level + 1)

    # Process files - with special handling for image-only folders
    if files:
        image_extensions = {".jpg", ".jpeg", ".png"}
        is_image_only_folder = all(f.suffix.lower() in image_extensions for f in files)

        files_to_display = files
        num_omitted = 0

        if is_image_only_folder and len(files) > max_files_to_list_in_image_folder:
            files_to_display = random.sample(files, max_files_to_list_in_image_folder)
            # Sorting the sample for consistent-looking output, though "random" implies order isn't key
            files_to_display.sort(key=lambda p: p.name.lower()) 
            num_omitted = len(files) - len(files_to_display)

        for i, path in enumerate(files_to_display):
            connector = "├── " if i < len(files_to_display) - 1 else "└── "
            report_lines.append(f"{prefix}{connector}[F] {path.name}")
        
        if num_omitted > 0:
            report_lines.append(f"{prefix}    (... and {num_omitted} more image files)")
    else: # No files to process or original 'files' list was empty
        pass # No files to list for this directory

    return report_lines


def generate_report(root_dir: Path) -> str:
    """
    Generates a comprehensive report of the directory structure and file types.
    """
    report_parts = []
    report_parts.append(f"Directory Structure Report for: {root_dir.resolve()}")
    report_parts.append("=" * (30 + len(str(root_dir.resolve()))))
    report_parts.append("")

    # Generate the tree structure
    tree_lines = print_tree(root_dir, report_lines=[f"[R] {root_dir.name}"])
    report_parts.extend(tree_lines)
    report_parts.append("")

    # --- Summary ---
    report_parts.append("--- File Type Summary ---")
    
    total_dirs = 0
    total_files = 0
    file_extension_counts = Counter()

    for item in root_dir.rglob("*"): # rglob includes all subdirectories
        # Check if the item or any of its parents are in IGNORED_DIRS
        if any(ignored_dir in item.parts for ignored_dir in IGNORED_DIRS if ignored_dir not in {"*.egg-info"}): # Handle glob patterns separately
            continue
        if any(item.match(pattern) for pattern in IGNORED_DIRS if "*" in pattern):
            continue

        if item.is_dir():
            total_dirs += 1
        elif item.is_file():
            total_files += 1
            ext = item.suffix.lower()
            if ext in COUNTED_EXTENSIONS:
                file_extension_counts[COUNTED_EXTENSIONS[ext]] += 1
            elif ext: # Count other known extensions
                file_extension_counts[f"Other Files ({ext})"] += 1
            else: # Files with no extension
                file_extension_counts["Files (no extension)"] += 1
                
    report_parts.append(f"Total Directories Scanned (excluding ignored): {total_dirs}")
    report_parts.append(f"Total Files Scanned (excluding ignored): {total_files}")
    
    if file_extension_counts:
        report_parts.append("\nFile Counts by Type:")
        for ext_type, count in sorted(file_extension_counts.items()):
            report_parts.append(f"  {ext_type}: {count}")
            
    return "\n".join(report_parts)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate a directory structure report for a given path."
    )
    parser.add_argument(
        "directory",
        type=str,
        nargs="?",
        default=".",
        help="The root directory to scan (defaults to the current directory).",
    )
    args = parser.parse_args()

    target_directory = Path(args.directory)

    if not target_directory.exists() or not target_directory.is_dir():
        print(f"Error: The specified path '{target_directory}' is not a valid directory.")
    else:
        report = generate_report(target_directory)
        print(report)