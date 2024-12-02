import os

def rename_seg_files(folder_path):
    """
    Rename files in the given folder by removing 'SEG' from the end of the filename
    (before the extension).
    
    Args:
        folder_path (str): Path to the folder containing files to rename
    """
    # Check if the folder exists
    if not os.path.exists(folder_path):
        print(f"Error: Folder '{folder_path}' does not exist")
        return
    
    # Counter for renamed files
    renamed_count = 0
    
    # Iterate through all files in the folder
    for filename in os.listdir(folder_path):
        # Split the filename into name and extension
        name, ext = os.path.splitext(filename)
        # Check if filename ends with 'SEG' before the extension
        if name.endswith('SEG'):
            # Create new filename by removing 'SEG'
            new_name = name[:-3] + ext
            
            # Full paths for old and new filenames
            old_path = os.path.join(folder_path, filename)
            new_path = os.path.join(folder_path, new_name)
            
            try:
                # Rename the file
                os.rename(old_path, new_path)
                print(f"Renamed: {filename} -> {new_name}")
                renamed_count += 1
            except OSError as e:
                print(f"Error renaming {filename}: {e}")
    
    # Print summary
    print(f"\nProcess completed. {renamed_count} files renamed.")

# Example usage
if __name__ == "__main__":
    # Replace this with your folder path
    folder_path = "C:\Main Folder\GDP\\nerf\pipeline\\360scene\\images_segmented"
    rename_seg_files(folder_path)