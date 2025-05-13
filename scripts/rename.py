import os
import re
import glob

def standardize_filename(filename):
    """
    Convert filenames from different formats:
    
    1. penny_2019_obverse_D_12179.jpg -> date_2019_obverse_D_12179.jpg
    """
    # Handle the penny_ format seen in your date folder
    if filename.startswith("penny_"):
        # This pattern matches: penny_YEAR_obverse_MINTMARK_ID.EXTENSION
        pattern = r'penny_(\d+)_obverse_([^_\.]+)_(\d+)\.(.+)'
        match = re.match(pattern, filename)
        
        if match:
            year, mint_mark, id_num, extension = match.groups()
            
            # Create the new filename with 'date' prefix
            new_filename = f"date_{year}_obverse_{mint_mark}_{id_num}.{extension}"
            return new_filename
    
    # If we get here, we couldn't parse the filename
    print(f"Warning: Could not parse filename: {filename}")
    return None

def rename_files_in_directory(directory, perform_rename=False):
    """Rename all matching files in the specified directory"""
    # Get all image files in the directory
    file_paths = glob.glob(os.path.join(directory, "*.png")) + glob.glob(os.path.join(directory, "*.jpg"))
    renamed_count = 0
    
    for file_path in file_paths:
        filename = os.path.basename(file_path)
        
        # Skip files that already start with 'date_'
        if filename.startswith("date_"):
            print(f"File already starts with 'date_': {filename}")
            continue
        
        # Process files that start with "penny_"
        if not filename.startswith("penny_"):
            print(f"Skipping file that doesn't start with 'penny_': {filename}")
            continue
            
        new_filename = standardize_filename(filename)
        if new_filename:
            new_path = os.path.join(os.path.dirname(file_path), new_filename)
            
            print(f"Renaming: {filename} -> {new_filename}")
            renamed_count += 1
            
            if perform_rename:
                # Check if destination file already exists
                if os.path.exists(new_path):
                    print(f"Warning: Destination file already exists: {new_path}")
                    # Option 1: Skip this file
                    print(f"Skipping rename of {filename}")
                    continue
                    
                    # Option 2 (alternative): Delete the existing file first
                    # print(f"Removing existing file: {new_path}")
                    # os.remove(new_path)
                
                os.rename(file_path, new_path)
    
    return renamed_count

# Directory to process (only the date folder)
directory = r"C:\Users\nickh\OneDrive\Documents\nh\Git_repo\coin_sorter3\data\date"

def main():
    print("Coin Image Filename Standardizer")
    print("="*40)
    
    # First run in preview mode
    perform_rename = False
    total_renamed = 0
    
    print(f"\nProcessing directory: {directory}")
    if os.path.exists(directory):
        count = rename_files_in_directory(directory, perform_rename)
        total_renamed += count
    else:
        print(f"Directory not found: {directory}")
    
    print("\n" + "="*40)
    print(f"Summary: {total_renamed} files would be renamed")
    
    if total_renamed > 0:
        response = input("\nDo you want to proceed with renaming these files? (yes/no): ").strip().lower()
        if response == 'yes':
            print("\nPerforming actual renames...")
            
            # Reset counter and run again with actual renaming
            total_renamed = 0
            if os.path.exists(directory):
                count = rename_files_in_directory(directory, perform_rename=True)
                total_renamed += count
            
            print(f"\nSuccessfully renamed {total_renamed} files.")
        else:
            print("\nRename operation cancelled. Exiting.")
    else:
        print("\nNo files need to be renamed. Exiting.")

if __name__ == "__main__":
    main()