import math
# from Queue import Queue
import queue
import time
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import torch
import sys
import threading
from pynput import mouse
import instant_nerf as nerf
import logging
from tqdm import tqdm

"""
Custom NeRF visualiser made by Shree Kandekar
"""

class InputController:
    button_pressed_pos = np.array([0,0]) # Position of the last mouse button push down
    change_on_drag = np.array([0,0]) # Change of the mouse button while the left mouse button is currently pressed
    diff = np.array([0,0]) # Difference from the last left mouse button up
    overall_diff = np.array([0,0]) # Overall change from origin
    left_button_pressed = False
    mouse_listener_thread = None

    def __init__(self):
        if InputController.mouse_listener_thread is None:
            # Start the mouse listener in a separate thread
            InputController.mouse_listener_thread = threading.Thread(target=self.__start_mouse_listener)
            InputController.mouse_listener_thread.daemon = True  # Ensures thread will close with the main program
            InputController.mouse_listener_thread.start()

    @classmethod
    def __on_move(cls, x, y):
        # global button_pressed_pos, diff, change_on_drag
        if cls.left_button_pressed:
            cls.change_on_drag = np.array([x, y]) - cls.button_pressed_pos
            cls.overall_diff = cls.diff + cls.change_on_drag
            logging.debug(f"Change on drag: {cls.change_on_drag}")


    # Callback function to capture mouse button presses
    @classmethod
    def __on_click(cls, x, y, button, pressed):
        # global left_button_pressed, button_pressed_pos, diff, change_on_drag  # Declare as global to modify outside function
        if button == mouse.Button.left:
            cls.left_button_pressed = pressed  # True if pressed, False if released
            cls.button_pressed_pos = np.array([x, y])
            if not cls.left_button_pressed:
                cls.diff = cls.diff = cls.change_on_drag
                cls.change_on_drag = np.array([0, 0])
            logging.debug("Left button pressed" if pressed else "Left button released")
    
    @classmethod
    def __start_mouse_listener(cls):
        with mouse.Listener(on_move=cls.__on_move, on_click=cls.__on_click) as listener:
            listener.join()

    @classmethod
    def stop_mouse_listener(cls, event):
        sys.exit(0)
        return False

class Visualiser:

    camera_pose_ax = None
    nerf_ax = None
    seg_ax = None
    sensitivity = 50 # Lower is more sensitive
    image_queue = queue.Queue() # Images to be rendered (should have max 1)
    N = 16 # Number of divisions when rendering nerf
    H = 200
    W = 200

    original_pose = None

    model = None # Nerf model
    cancel_prev_render = threading.Event()
    cancel_prev_render.clear()

    def __init__(self, initial_pose, model_name, num_of_labels):
        plt.ion()  # Turn on interactive mode
        fig = plt.figure(figsize=(10, 7))
        fig.canvas.mpl_connect('close_event', InputController.stop_mouse_listener)

        Visualiser.camera_pose_ax = fig.add_subplot(131, projection='3d')
        Visualiser.nerf_ax = fig.add_subplot(132)
        Visualiser.seg_ax = fig.add_subplot(133)
        Visualiser.pbar = tqdm(total=Visualiser.N, desc="Processing")

        Visualiser.original_pose = initial_pose
        Visualiser.model = nerf.load_model(model_name, num_of_labels)
        # Disable interactive rotation
        Visualiser.camera_pose_ax.disable_mouse_rotation()
        Visualiser.render_loop()


    @staticmethod
    def rotate_z(matrix, theta):
        """
        Rotate a 4x4 transformation matrix around the z-axis by angle theta.
        """

        # Define the rotation matrix around the z-axis
        R_z = np.array([
            [np.cos(theta), -np.sin(theta), 0, 0],
            [np.sin(theta),  np.cos(theta), 0, 0],
            [0,             0,             1, 0],
            [0,             0,             0, 1]
        ])
        
        # Perform matrix multiplication to get the rotated matrix
        rotated_matrix = np.dot(R_z, matrix)

        return rotated_matrix
    
    @staticmethod
    def rotate_y(matrix, theta):
        """
        Rotate a 4x4 transformation matrix around the y-axis by angle theta.
        """
        # Define the rotation matrix around the z-axis
        R_y = np.array([
                [np.cos(theta), 0, np.sin(theta), 0],
                [0, 1, 0, 0],
                [-np.sin(theta), 0, np.cos(theta), 0],
                [0, 0, 0, 1]
            ])
        
        # Perform matrix multiplication to get the rotated matrix
        rotated_matrix = np.dot(R_y, matrix)

        return rotated_matrix

    @staticmethod
    def rotate_latitude(matrix, theta):
        """
        Rotates the view up and down
        """
        current_z_angle = np.arctan2(matrix[1, 3], matrix[0, 3])
        
        # Combine the rotations
        rotated_matrix = Visualiser.rotate_z(Visualiser.rotate_y(Visualiser.rotate_z(matrix, -current_z_angle), theta), current_z_angle)
        return rotated_matrix

    @classmethod
    def plot_rotation_matrix(cls, R):
        """
        Plots camera pose vectors
        """

        origin = R[:, 3]

        cls.camera_pose_ax.scatter([0], [0], [0], marker='o')

        for i, (c, label) in enumerate(zip(['r', 'g', 'b'], ['Right', 'Up', 'Forward'])):
            cls.camera_pose_ax.quiver(origin[0], origin[1], origin[2],
                    R[0,i], R[1,i], R[2,i],
                    color=c)
    
    @classmethod
    def calculate_new_pose(cls):
        theta_long = math.radians(InputController.overall_diff[0]/cls.sensitivity)
        theta_lat = math.radians(InputController.overall_diff[1]/cls.sensitivity)
        pose = cls.rotate_latitude(cls.original_pose, theta_lat)
        pose = cls.rotate_z(pose, theta_long)
        # Visualiser.plot_rotation_matrix(pose)
        return pose

    @classmethod
    def render_all_camera_poses(cls, poses):
        """
        Renders all the camera poses
        """
        cls.camera_pose_ax.set_xlim([-4,4])
        cls.camera_pose_ax.set_ylim([-4,4])
        cls.camera_pose_ax.set_zlim([-4,4])
        
        # Labels
        cls.camera_pose_ax.set_xlabel('X')
        cls.camera_pose_ax.set_ylabel('Y')
        cls.camera_pose_ax.set_zlabel('Z')
        cls.camera_pose_ax.legend()

        cls.camera_pose_ax.view_init(elev=30, azim=60)
        for ind, pose in enumerate(poses):
            cls.plot_rotation_matrix(pose)
    
    @classmethod
    def render_cur_camera_pose(cls, pose):
        """
        Renders the current camera pose
        """
        cls.camera_pose_ax.set_xlim([-4,4])
        cls.camera_pose_ax.set_ylim([-4,4])
        cls.camera_pose_ax.set_zlim([-4,4])
        
        # Labels
        cls.camera_pose_ax.set_xlabel('X')
        cls.camera_pose_ax.set_ylabel('Y')
        cls.camera_pose_ax.set_zlabel('Z')
        cls.camera_pose_ax.legend()

        cls.camera_pose_ax.view_init(elev=30, azim=60)

        theta_long = math.radians(InputController.overall_diff[0]/cls.sensitivity)
        theta_lat = math.radians(InputController.overall_diff[1]/cls.sensitivity)
        pose = cls.rotate_latitude(pose, theta_lat)
        pose = cls.rotate_z(pose, theta_long)
        cls.plot_rotation_matrix(pose)
    
    @classmethod
    def render_nerf_image(cls):
        """
        Renders NeRF image
        """
        if not cls.image_queue.empty():
            latest_img, ax = cls.image_queue.get()
            ax.imshow(latest_img)

    @classmethod
    def create_nerf_renderer(cls, pose):
        """
        Creates a generator for nerf output
        """
        return nerf.get_output_for_img_iter(cls.model, hn=nerf.HN, hf=nerf.HF, nb_bins=nerf.NB_BINS, 
                                            testpose=torch.from_numpy(pose).float(), H=cls.H, W=cls.W, focal=1657, N=cls.N,
                                            batch_size=nerf.batch_size, flag=cls.cancel_prev_render, pbar=cls.pbar)
    
    @classmethod
    def run_nerf_renderer(cls, renderer):
        """
        Runs generator
        """
        for img, seg_img in renderer:
            # if not cls.image_queue.empty():
            #     cls.image_queue.get_nowait()  # Remove previous image if exists
            cls.image_queue.put((img, cls.nerf_ax))
            if seg_img is not None:
                cls.image_queue.put((seg_img, cls.seg_ax))

    @classmethod
    def render_loop(cls):
        """
        Main render loop
        """
        last_render = np.array([0,0])

        nerf_thread = None
        try:
            while True:
                moved = np.linalg.norm(last_render - InputController.overall_diff) > 1

                if moved or nerf_thread is None:
                    last_render = InputController.overall_diff

                    cls.cancel_prev_render.set()
                    if nerf_thread is not None:
                        nerf_thread.join()

                    cls.camera_pose_ax.clear()
                    cls.cancel_prev_render.clear()

                    pose = cls.calculate_new_pose()
                    cls.render_cur_camera_pose(pose)
                    nerf_gen = cls.create_nerf_renderer(pose)
                    nerf_thread = threading.Thread(target=cls.run_nerf_renderer, args=(nerf_gen, ), daemon=True)
                    nerf_thread.start()
                
                cls.render_nerf_image()
            
                plt.pause(0.01)
        except KeyboardInterrupt:
            InputController.stop_mouse_listener()
            print("Image rendering stopped.")


loaded = np.load("nerf_formated_data.npz")
poses = torch.from_numpy(loaded['poses_train'])

model = 'model_state_dict_15.pth'

logging.basicConfig(level=logging.INFO)

InputController()
Visualiser(poses[0], model, num_of_labels=4)