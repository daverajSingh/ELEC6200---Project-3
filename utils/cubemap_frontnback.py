import cv2
import os
import numpy as np
import argparse

# Constants for cubemap face generation
CUBE_FACE_SIZE = 512  # Size of each face image in the cubemap

def load_equirectangular_images(input_dir):
    """
    Loads all equirectangular images from the specified directory.
    """
    image_files = sorted([f for f in os.listdir(input_dir) if f.endswith('.jpg') or f.endswith('.png')])
    equirectangular_images = [cv2.imread(os.path.join(input_dir, img_file)) for img_file in image_files]
    return equirectangular_images, image_files

def equirectangular_to_cubemap(equi_img):
    """
    Convert an equirectangular image into two cubemap faces (front and back).
    """
    face_size = CUBE_FACE_SIZE
    faces = {}

    # Define angles for the front and back faces
    phi_angles = {
        'front': 0,          # Front face at 0 degrees
        'back': np.pi        # Back face at 180 degrees
    }
    
    # Generate front and back faces only
    for face_name, phi in phi_angles.items():
        face_img = generate_cubemap_face(equi_img, phi, face_size)
        faces[face_name] = face_img

    return faces

def generate_cubemap_face(equi_img, phi, face_size):
    """
    Generate a single cubemap face from an equirectangular image given a specific phi (longitude).
    Uses spherical-to-Cartesian conversion with cv2.remap.
    """
    h, w = equi_img.shape[:2]
    theta = np.pi / 2  # Constant for 90 degrees to center vertically

    # Create mesh grid for each cubemap face
    x = np.linspace(-1, 1, face_size)
    y = np.linspace(-1, 1, face_size)
    x_grid, y_grid = np.meshgrid(x, y)

    # Convert 2D cubemap coordinates to 3D unit vectors
    z_grid = np.ones_like(x_grid)
    direction_vectors = np.stack([x_grid, y_grid, z_grid], axis=-1)
    direction_vectors /= np.linalg.norm(direction_vectors, axis=-1, keepdims=True)

    # Convert Cartesian coordinates to spherical coordinates (theta, phi)
    theta_map = np.arcsin(direction_vectors[..., 1]) + theta  # Latitude
    phi_map = np.arctan2(direction_vectors[..., 0], direction_vectors[..., 2]) + phi  # Longitude

    # Normalize theta and phi to match equirectangular coordinates
    u_map = (phi_map / (2 * np.pi) + 0.5) * w
    v_map = (theta_map / np.pi) * h

    # Convert u_map and v_map to float32 for cv2.remap compatibility
    u_map = u_map.astype(np.float32)
    v_map = v_map.astype(np.float32)

    # Remap equirectangular image to the cubemap face
    face_img = cv2.remap(equi_img, u_map, v_map, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)
    return face_img

def save_cubemap_faces(cubemap_faces, output_dir, image_name):
    """
    Saves only the 'front' and 'back' cubemap faces as JPEG images.
    """
    os.makedirs(output_dir, exist_ok=True)
    base_name = os.path.splitext(image_name)[0]
    for face_name, face_img in cubemap_faces.items():
        output_path = os.path.join(output_dir, f"{base_name}_{face_name}.jpg")
        cv2.imwrite(output_path, face_img)
        print(f"Saved {output_path}")

def process_equirectangular_images(input_dir, output_dir):
    """
    Main function to process each equirectangular image into front and back cubemap faces.
    """
    equirectangular_images, image_files = load_equirectangular_images(input_dir)

    for img_file, equi_img in zip(image_files, equirectangular_images):
        cubemap_faces = equirectangular_to_cubemap(equi_img)
        save_cubemap_faces(cubemap_faces, output_dir, img_file)

if __name__ == "__main__":
    # Argument parser setup
    parser = argparse.ArgumentParser(description="Process equirectangular images into cubemap faces.")
    parser.add_argument("-i", "--input_dir", required=True, help="Input directory containing equirectangular images")
    parser.add_argument("-o", "--output_dir", required=True, help="Output directory to save cubemap faces")

    args = parser.parse_args()
    
    # Process the images
    process_equirectangular_images(args.input_dir, args.output_dir)
