# refactor_metadata_hardcoded.py
# Purpose: Refactors an existing penny image metadata JSON to a new standardized schema.
# Filenames and UIDs are preserved. Year is stored as a string.
# Assumes input filenames are QA'd.
# Author: Your Name / AI Assistant
# Date: 2025-05-16

import json
import os
import re
from pathlib import Path

# --- Configuration ---
# Hardcoded path for the input metadata file
INPUT_METADATA_PATH = Path(r"C:\Users\nickh\OneDrive\Documents\nh\Git_repo\coin_sorter3\metadata\metadata.json")
# Output file will be saved in the same directory as the input file
OUTPUT_METADATA_PATH = INPUT_METADATA_PATH.parent / "metadata_standardized.json"

# --- Mappings ---
REVERSE_MAP = {
    'b': 'bicentennial',
    'm': 'memorial',
    'w': 'wheat',
    's': 'shield'
}

# --- Helper Function for Filename Parsing ---
def parse_filename_details(filename: str, original_entry: dict = None) -> dict:
    """
    Parses the filename to extract relevant details.
    Relies on filenames being QA'd and conforming to expected patterns.
    """
    name_part, _ = os.path.splitext(filename)
    parts = name_part.split('_')

    parsed_info = {
        "filename_prefix": None,
        "year_str": None,
        "side_from_filename": None,
        "mint_str": "nomint", # Default
        "uid_str": None,
        "reverse_code": None
    }

    if not parts:
        return parsed_info

    parsed_info["filename_prefix"] = parts[0].lower()

    # Common pattern: prefix_year_side_mint_uid (e.g., date_1923_obverse_S_3001.jpg, penny_1945_obverse_D_1001.jpg)
    if len(parts) >= 5 and parts[0].lower() in ["penny", "date", "mint", "liberty"] and parts[2].lower() in ["obverse", "front"]:
        parsed_info["year_str"] = parts[1]
        parsed_info["side_from_filename"] = "obverse"
        parsed_info["mint_str"] = parts[3] if parts[3] else "nomint"
        parsed_info["uid_str"] = parts[4]
    # Pattern: penny_reverse_CODE_UID (e.g., penny_reverse_w_2001.jpg)
    elif parts[0].lower() == "penny" and len(parts) >= 4 and parts[1].lower() == "reverse":
        parsed_info["side_from_filename"] = "reverse"
        parsed_info["reverse_code"] = parts[2].lower()
        parsed_info["uid_str"] = parts[3]
    # Older pattern: penny_YEAR_UID_M (less specific, might need data from original_entry)
    elif parts[0].lower() == "penny" and len(parts) == 4 and parts[3].upper() == 'M': # penny_1960_1234_M
        parsed_info["year_str"] = parts[1]
        parsed_info["uid_str"] = parts[2]
        parsed_info["side_from_filename"] = "obverse" # Assume obverse
        # Mint could be 'M' or derived from original_entry if more specific
        if original_entry and original_entry.get("mint"):
             parsed_info["mint_str"] = original_entry.get("mint")
        else:
            parsed_info["mint_str"] = "nomint" # Default or M if M is a valid mint mark here
    # Simpler ROI: roi-type_year_uid (assuming obverse, nomint)
    elif parts[0].lower() in ["date", "mint", "liberty"] and len(parts) == 3:
        parsed_info["year_str"] = parts[1]
        parsed_info["uid_str"] = parts[2]
        parsed_info["side_from_filename"] = "obverse" # Default for these ROIs
        parsed_info["mint_str"] = "nomint" # Default
    else: # Fallback for UID if other parsing failed
        if original_entry and original_entry.get("uid"):
            parsed_info["uid_str"] = str(original_entry.get("uid"))
        elif len(parts) > 1: # Try to get the last part as UID if it's numeric
            if parts[-1].isdigit():
                parsed_info["uid_str"] = parts[-1]

    # Ensure year and UID are strings if found
    if parsed_info["year_str"]:
        parsed_info["year_str"] = str(parsed_info["year_str"])
    if parsed_info["uid_str"]:
        parsed_info["uid_str"] = str(parsed_info["uid_str"])
    if not parsed_info["mint_str"]: # Ensure mint_str is not None
        parsed_info["mint_str"] = "nomint"


    return parsed_info

# --- Main Refactoring Logic ---
def refactor_metadata(input_path: Path, output_path: Path) -> None:
    if not input_path.exists():
        print(f"Error: Input metadata file not found at {input_path}")
        return

    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            old_metadata_list = json.load(f)
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {input_path}. File might be corrupted.")
        return

    new_metadata_list = []
    print(f"Starting refactoring of {len(old_metadata_list)} entries from {input_path}...")

    for old_entry in old_metadata_list:
        new_entry = {}
        filename = old_entry.get("filename")

        if not filename:
            print(f"Warning: Skipping entry due to missing filename: {old_entry}")
            continue

        new_entry["filename"] = filename
        parsed_details = parse_filename_details(filename, old_entry)

        # Use UID from parsing, fall back to old entry's UID, then to UNKNOWN_UID
        new_entry["uid"] = parsed_details.get("uid_str") or str(old_entry.get("uid", "UNKNOWN_UID"))

        original_type_field = old_entry.get("type", "").lower() # e.g. "liberty", "mint", "date", "obverse", "reverse"
        filename_prefix = parsed_details.get("filename_prefix")

        if filename_prefix in ["date", "mint", "liberty"] or original_type_field in ["date", "mint", "liberty"]:
            new_entry["type"] = "roi"
            new_entry["side"] = "obverse"
            new_entry["roi_type"] = filename_prefix if filename_prefix in ["date", "mint", "liberty"] else original_type_field
            new_entry["year"] = str(parsed_details.get("year_str") or old_entry.get("year", "UNKNOWN_YEAR"))
            new_entry["mint"] = parsed_details.get("mint_str") or old_entry.get("mint", "nomint")
        elif filename_prefix == "penny":
            new_entry["type"] = "penny"
            if parsed_details.get("side_from_filename") == "reverse":
                new_entry["side"] = "reverse"
                reverse_code = parsed_details.get("reverse_code")
                new_entry["reverse_type"] = REVERSE_MAP.get(reverse_code, reverse_code or "unknown_reverse")
            else: # Assume obverse
                new_entry["side"] = "obverse"
                new_entry["year"] = str(parsed_details.get("year_str") or old_entry.get("year", "UNKNOWN_YEAR"))
                new_entry["mint"] = parsed_details.get("mint_str") or old_entry.get("mint", "nomint")
        elif original_type_field == "obverse":
            new_entry["type"] = "penny"
            new_entry["side"] = "obverse"
            new_entry["year"] = str(old_entry.get("year") or parsed_details.get("year_str", "UNKNOWN_YEAR"))
            new_entry["mint"] = old_entry.get("mint") or parsed_details.get("mint_str", "nomint")
        elif original_type_field == "reverse":
            new_entry["type"] = "penny"
            new_entry["side"] = "reverse"
            if "reverse_type" in old_entry: # Prefer existing mapped field
                 new_entry["reverse_type"] = old_entry["reverse_type"]
            elif parsed_details.get("reverse_code"):
                 new_entry["reverse_type"] = REVERSE_MAP.get(parsed_details["reverse_code"], parsed_details["reverse_code"])
            else:
                 new_entry["reverse_type"] = "unknown_reverse"
        else:
            print(f"Warning: Could not confidently determine type for '{filename}' (Old type: '{original_type_field}'). Assigning default 'unknown'.")
            new_entry["type"] = "unknown"
            new_entry["side"] = "unknown"
            # Try to populate other fields if possible from old entry
            if "year" in old_entry: new_entry["year"] = str(old_entry["year"])
            if "mint" in old_entry: new_entry["mint"] = old_entry["mint"]
            if "roi_type" in old_entry: new_entry["roi_type"] = old_entry["roi_type"]
            if "reverse_type" in old_entry: new_entry["reverse_type"] = old_entry["reverse_type"]


        # Final check for year string conversion
        if "year" in new_entry and not isinstance(new_entry["year"], str):
            new_entry["year"] = str(new_entry["year"])
        if "mint" in new_entry and (not new_entry["mint"] or new_entry["mint"].lower() == "none"):
            new_entry["mint"] = "nomint"


        new_metadata_list.append(new_entry)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(new_metadata_list, f, indent=2)

    print(f"Successfully refactored {len(new_metadata_list)} entries.")
    print(f"New metadata saved to: {output_path}")

if __name__ == "__main__":
    print("Starting metadata refactoring process...")
    if not INPUT_METADATA_PATH.exists():
        print(f"CRITICAL ERROR: Input metadata file not found at the hardcoded path: {INPUT_METADATA_PATH}")
        print("Please ensure the path is correct in the script.")
    else:
        refactor_metadata(INPUT_METADATA_PATH, OUTPUT_METADATA_PATH)
    print("Refactoring complete.")