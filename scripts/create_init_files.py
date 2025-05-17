# create_init_files.py
# A simple script to create necessary __init__.py files.

import os

paths = [
    "src/__init__.py",
    "src/model_training/__init__.py",
    "tests/__init__.py",
    "tests/model_training/__init__.py",
    # Add any other __init__.py paths if needed
]

for file_path in paths:
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "a") as f: # 'a' to create if not exists, without truncating
        pass # Creates an empty file
    print(f"Created: {file_path}")

print("Done creating __init__.py files.")