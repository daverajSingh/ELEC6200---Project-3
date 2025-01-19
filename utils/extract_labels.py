import math
import os
import numpy as np
from PIL import Image
import colorsys

MAX_DISTANCE = 25 # Max distance between colors 


def calc_distance(c1, c2):
    # c1 = np.array(c1, dtype=np.int16)
    # c2 = np.array(c2, dtype=np.int16)
    # # print(c1, c2, np.linalg.norm(c2 - c1, axis=-1), c2 - c1)
    # return np.linalg.norm(c2 - c1, axis=-1)
    # c1 /= float(256)
    # c /= float(256)
    # print(c1)
    # print(colorsys.rgb_to_hsv(*c1)[0])
    # print(colorsys.rgb_to_hsv(*c1)[0], colorsys.rgb_to_hsv(*c2)[0])
    # return abs(colorsys.rgb_to_hsv(*c1)[0] - colorsys.rgb_to_hsv(*c2)[0])

    # c1 = np.array([c1[0], c1[2]], dtype=np.int16)
    # c2 = np.array([c2[0], c2[2]], dtype=np.int16)
    # return np.linalg.norm(c2 - c1, axis=-1)

    return abs(int(c1[0]) - int(c2[0]))

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
    colors_to_count = {(0,0,0): 0}
    
    for filename in os.listdir(folder_path):
        if filename.endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff')):
            # Full path to the image
            file_path = os.path.join(folder_path, filename)
            
            # Open the image
            img = Image.open(file_path).convert('HSV')
            img_array = np.array(img)
            
            
            # Find unique colors and add to global set
            unique_colors = [tuple(color) for color in np.unique(img_array.reshape(-1, img_array.shape[-1]), axis=0)]

            for c in unique_colors:
                if c[2] < 50:
                    colors_to_count[(0,0,0)] += 1
                    continue

                already_present = False
                for c2 in global_unique_colors:
                    if calc_distance(c, c2) <= MAX_DISTANCE:
                        already_present = True
                        colors_to_count[c2] += 1
                        break
                if not already_present:
                    colors_to_count[c] = 1
                    global_unique_colors.add(c)

    # Create global label mapping
    global_label_mapping = {(0,0,0): 0}

    for label_id, color in enumerate(global_unique_colors - {(0,0,0)}):
        if colors_to_count[color]/len(os.listdir(folder_path)) < 10:
            continue
        global_label_mapping[color] = label_id + 1

    return global_label_mapping

def process_image_with_global_mapping(img, global_label_mapping):
    # Convert image to numpy array
    img_array = np.array(img, dtype=np.int16)
    
    # Create label ID array using global mapping
    label_id_array = np.zeros_like(img_array[:,:,0], dtype=int)
    distances_array = np.full_like(img_array[:, :, 0], np.inf, dtype=float)
    for color, label_id in global_label_mapping.items():
        diff = np.abs(img_array - np.array(color, dtype=np.int16))  # Broadcast subtraction
        # distances = np.linalg.norm(diff, axis=-1)  # Compute Euclidean distance
        distances = diff[:,:,0]
        mask = distances < distances_array
        print(label_id, np.sum(mask))
        distances_array[mask] = distances[mask]
        label_id_array[mask] = label_id        
    
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
            img = Image.open(file_path).convert('HSV')
            
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