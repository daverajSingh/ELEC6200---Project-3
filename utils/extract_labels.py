import os
import numpy as np
from PIL import Image

MAX_DISTANCE = 150 # Max distance between colors 


def calc_distance(c1, c2):
    c1 = np.array(c1, dtype=np.int16)
    c2 = np.array(c2, dtype=np.int16)
    # print(c1, c2, np.linalg.norm(c2 - c1, axis=-1), c2 - c1)
    return np.linalg.norm(c2 - c1, axis=-1)

def create_global_label_mapping(folder_path):
    """
    Create a global label mapping by finding all unique colors across all images.
    
    Args:
        folder_path (str): Path to the folder containing segmentation images
    
    Returns:
        dict: A mapping of unique colors to global label IDs
    """
    # Collect all unique colors across all images
    global_unique_colors = set([(0,0,0)])
    
    for filename in os.listdir(folder_path):
        if filename.endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff')):
            # Full path to the image
            file_path = os.path.join(folder_path, filename)
            
            # Open the image
            img = Image.open(file_path)
            
            # Convert image to numpy array
            img_array = np.array(img)
            
            # Find unique colors and add to global set
            unique_colors = [tuple(color) for color in np.unique(img_array.reshape(-1, img_array.shape[-1]), axis=0)]

            for c in unique_colors:
                already_present = False
                for c2 in global_unique_colors:
                    if calc_distance(c, c2) <= MAX_DISTANCE:
                        already_present = True
                        break
                if not already_present:
                    global_unique_colors.add(c)

    # Create global label mapping
    global_label_mapping = {(0,0,0): 0}

    for label_id, color in enumerate(global_unique_colors - {(0,0,0)}):
        global_label_mapping[color] = label_id + 1

    return global_label_mapping

def process_image_with_global_mapping(img, global_label_mapping):
    # Convert image to numpy array
    img_array = np.array(img, dtype=np.int16)
    
    # Create label ID array using global mapping
    label_id_array = np.zeros_like(img_array[:,:,0], dtype=int)
    for color, label_id in global_label_mapping.items():
        # mask = np.all(img_array == color, axis=-1)
        # label_id_array[mask] = label_id

        diff = img_array - np.array(color, dtype=np.int16)  # Broadcast subtraction
        distances = np.linalg.norm(diff, axis=-1)  # Compute Euclidean distance
        label_id_array[distances <= MAX_DISTANCE] = label_id
    
    return label_id_array

def process_images_with_global_mapping(folder_path, global_label_mapping):
    """
    Process images using a global label mapping.
    
    Args:
        folder_path (str): Path to the folder containing segmentation images
        global_label_mapping (dict): Mapping of colors to global label IDs
    
    Returns:
        dict: A dictionary with image filenames and their processed label information
    """
    # Store results
    results = {}
    
    # Iterate through all image files in the folder
    for filename in os.listdir(folder_path):
        if filename.endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff')):
            # Full path to the image
            file_path = os.path.join(folder_path, filename)
            
            # Open the image
            img = Image.open(file_path)
            
            # Create label ID array using global mapping
            label_id_array = process_image_with_global_mapping(img, global_label_mapping)
            
            # Store results
            results[filename] = {
                'label_id_array': label_id_array
            }
    
    return results

def main(SEG_PATH):

    # Create global label mapping
    global_label_mapping = create_global_label_mapping(SEG_PATH)
    print("Global Label Mapping:")
    for color, label_id in global_label_mapping.items():
        print(f"Color {color}: Label ID {label_id}")
    
    segmentation_results = process_images_with_global_mapping(SEG_PATH, global_label_mapping)

    for filename, data in segmentation_results.items():
        print(f"\nFile: {filename}")
        print(f"Label ID Array Shape: {data['label_id_array'].shape}")
    
    return segmentation_results

if __name__ == "__main__":
    main()