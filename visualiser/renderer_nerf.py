import instant_nerf as nerf
from tqdm import tqdm
import threading
import queue
import torch
import numpy as np
import matplotlib.pyplot as plt

class Visualiser:
    camera_pose_ax = None
    nerf_ax = None
    seg_ax = None
    sensitivity = 5 # Lower is more sensitive
    image_queue = queue.Queue() # Images to be rendered (should have max 1)
    N = 16 # Number of divisions when rendering nerf
    H = 224
    W = 224
    # H = 1080 // 5
    # W = 1980 // 5

    num_of_labels = 4

    focal = None

    original_pose = None

    model = None # Nerf model

    cancel_prev_render = threading.Event()
    cancel_prev_render.clear()
    render_thread = None

    def __init__(self, model_name, num_of_labels, focal, initial_pose=None):
        Visualiser.pbar = tqdm(total=Visualiser.N, desc="Processing")
        Visualiser.model = nerf.load_model(model_name, num_of_labels)
        Visualiser.latest_img = None
        Visualiser.latest_seg = None
        Visualiser.focal = focal
        Visualiser.initial_pose = initial_pose
    
    @classmethod
    def load_model(cls, model):
        Visualiser.model = nerf.load_model(model, Visualiser.num_of_labels)
    
    @classmethod
    def is_model_loaded(cls):
        if cls.model is not None:
            return True

        return False

    @classmethod
    def get_nerf_image(cls):
        """
        Renders NeRF image
        """
        while not cls.image_queue.empty():
            img, t = cls.image_queue.get()
            if t == "IMG":
                Visualiser.latest_img = np.flip(img, 1) * 255
            elif t == "SEG":
                # Define a color map for the 5 classes (for example, use predefined RGB colors)
                color_map = [
                    [50, 50, 50],      # Red for class 0
                    [0, 255, 0],      # Green for class 1
                    [255, 0, 255],      # Blue for class 2
                    [255, 255, 0],    # Yellow for class 3
                    [0, 255, 255],    # Cyan for class 4
                ]

                # Initialize an RGB array (H, W, 3)
                rgb_array = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)

                # Map class indices to corresponding colors
                for i in range(5):  # Iterate over the 5 possible classes
                    rgb_array[img == i] = color_map[i]
                
                Visualiser.latest_seg = np.flip(rgb_array, 1)
    
                # cmap = plt.get_cmap('viridis')  # You can choose different colormaps like 'viridis', 'plasma', etc.
                # # normalized_array = (img - np.min(img)) / (np.max(img) - np.min(img))  # Normalize to [0, 1]
                # normalized_array = img / 5
                # print(normalized_array)
                # rgb_array = cmap(normalized_array)  # Get RGB values
                # Visualiser.latest_seg = (rgb_array * 255).astype(np.uint8)
                # Visualiser.latest_seg = np.repeat((img/5)[:, :, np.newaxis], 3, axis=2) * 255

        return Visualiser.latest_img, Visualiser.latest_seg
    
    @classmethod
    def run_nerf_renderer(cls, renderer):
        """
        Runs generator
        """
        for img, seg_img in renderer:
            # if not cls.image_queue.empty():
            #     cls.image_queue.get_nowait()  # Remove previous image if exists
            cls.image_queue.put((img, "IMG"))
            if seg_img is not None:
                cls.image_queue.put((seg_img, "SEG"))

    @classmethod
    def create_nerf_renderer(cls, pose):
        """
        Creates a generator for nerf output
        """
        # pose= cls.initial_pose
        return nerf.get_output_for_img_iter(cls.model, hn=nerf.HN, hf=nerf.HF, nb_bins=nerf.NB_BINS, 
                                            testpose=torch.from_numpy(pose).float(), H=cls.H, W=cls.W, focal=cls.focal, N=cls.N,
                                            batch_size=nerf.batch_size, flag=cls.cancel_prev_render, pbar=cls.pbar)

