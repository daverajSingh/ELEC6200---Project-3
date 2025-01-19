import os
import json
import numpy as np
import utils.extract_labels as extract_labels
from PIL import Image
# from sklearn.model_selection import train_test_split


def find_pose_key(img_filename, poses_list):
    """
    Find the corresponding pose key for an image filename
    where the key contains the image filename as a substring.
    
    Parameters:
    img_filename: str, name of the image file without extension
    poses_dict: dict, dictionary containing pose data
    
    Returns:
    str or None: matching key if found, None otherwise
    """
    for poses_dict in poses_list:
        if img_filename in poses_dict["file_path"]:
            return poses_dict
    return None

def load_data(image_folder_path, seg_image_folder_path, pose_json_path, test_size=None, random_state=42):
    """
    Load images and poses from specified paths and split into training and test sets.
    
    Parameters:
    image_folder_path: str, path to folder containing images
    pose_json_path: str, path to JSON file containing poses
    test_size: float, proportion of data to use for testing (default: 0.2)
    random_state: int, random seed for reproducibility (default: 42)
    
    Returns:
    dict containing training and test data
    """
    # Load poses from JSON
    with open(pose_json_path, 'r') as f:
        poses_list = json.load(f)["frames"]
    
    print("test 1")
    # global_label_mapping = extract_labels.create_global_label_mapping(seg_image_folder_path)
    global_label_mapping = {(0, 0, 0): 0, (231, 226, 220): 1, (29, 107, 153): 2}
    print(f"Number of labels: {len(global_label_mapping)}", global_label_mapping)

    # Load images
    images = []
    seg_images = []
    valid_poses = []
    image_files = sorted(os.listdir(image_folder_path))
    W = None
    H = None
    for img_file in image_files:
        if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
            print(img_file)
            img_path = os.path.join(image_folder_path, img_file)
            name, ext = os.path.splitext(img_file)
            seg_path = os.path.join(seg_image_folder_path, f"{name}SEG{ext}")
            try:
                # Load and convert image to numpy array
                img = Image.open(img_path)
                seg_img = Image.open(seg_path)
                img_array = np.array(img, dtype=np.float32) / 255.0
                H, W, _ = img_array.shape
                # Get corresponding pose (assuming filename matches pose key)
                img_key = f"{name[1:]}{ext}"
                pose_dict = find_pose_key(img_key, poses_list)

                if pose_dict is not None:
                    images.append(img_array)
                    valid_poses.append(pose_dict["transform_matrix"])

                    seg_images.append(extract_labels.process_image_with_global_mapping(seg_img, global_label_mapping))
                else:
                    raise Exception(f"Pose not found for image: {img_file}")
            except Exception as e:
                print(f"Error loading image {img_file}: {e}")
    
    print("test")
    images = np.array(images, dtype=np.float32)
    poses = np.array(valid_poses, dtype=np.float32)
    seg_images = np.array(seg_images, dtype=np.float32)
    
    return {
        'images_train': images,
        'seg_images': seg_images,
        'poses_train': poses,
        'W': W,
        'H': H
    }



def get_value_from_txt(txt_file_path):
    with open(txt_file_path, 'r') as f:
        # Get the 4th line (index 3 since zero-based)
        for i, line in enumerate(f):
            if i == 3:  # 4th line
                # Split by spaces and get 4th value (index 3)
                values = line.strip().split()
                focal = float(values[4])
                return focal
    return None



def main(image_dir, seg_image_dir, pose_path, camera_path, output_path='nerf_formated_data.npz'):
    focal = get_value_from_txt(camera_path)
    # Example usage:
    data = load_data(
        image_folder_path=image_dir,
        pose_json_path=pose_path,
        seg_image_folder_path=seg_image_dir
    )

    images_train = data['images_train']
    seg_images = data['seg_images']
    poses_train = data['poses_train']
    W = data['W']
    H = data['H']
    print(images_train.shape, seg_images.shape, poses_train.shape)

    np.savez(output_path, images_train=images_train, seg_images=seg_images, poses_train=poses_train, W=W, H=H, focal=focal)

if __name__=='__main__':
    CAMERAS_DIR = "colmap_output/colmap_text/cameras.txt"
    IMAGE_DIR = "images"
    POSE_PATH = 'colmap_output/transforms.json'
    main(IMAGE_DIR, POSE_PATH, CAMERAS_DIR)

# # Load the arrays
# loaded = np.load('arrays.npz')