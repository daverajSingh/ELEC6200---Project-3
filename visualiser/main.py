import queue
import random
import glfw
import OpenGL.GL as gl
from imgui.integrations.glfw import GlfwRenderer
import imgui
import numpy as np
import util
import imageio
import util_gau
import tkinter as tk
from tkinter import filedialog
import os
import sys
import argparse
from renderer_ogl import OpenGLRenderer, GaussianRenderBase
import time
import ctypes
from util import SimpleImageRenderer
import torch
import instant_nerf as nerf
import threading
from tqdm import tqdm
from renderer_nerf import Visualiser

# Add the directory containing main.py to the Python path
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(dir_path)

# Change the current working directory to the script's directory
os.chdir(os.path.dirname(os.path.abspath(__file__)))

def setup_cameras():
    g_camera_3dgs = util.Camera(1000, 1000)
    g_camera_nerf = util.Camera(1000, 1000)

    return g_camera_3dgs, g_camera_nerf

g_camera_3dgs, g_camera_nerf = setup_cameras()

BACKEND_OGL=0
BACKEND_CUDA=1
g_renderer_list = [
    None, # ogl
]
g_renderer_idx = BACKEND_OGL
g_renderer: GaussianRenderBase = g_renderer_list[g_renderer_idx]
g_scale_modifier = 1.
g_auto_sort = False
g_show_control_win = True
g_show_help_win = True
g_show_camera_win = False
g_render_mode_tables = ["Gaussian Ball", "Flat Ball", "Billboard", "Depth", "SH:0", "SH:0~1", "SH:0~2", "SH:0~3 (default)"]
g_render_mode = 7

update_3dgs_camera = True
update_nerf_camera = True

def impl_glfw_init():
    window_name = "NeUVF editor"

    if not glfw.init():
        print("Could not initialize OpenGL context")
        exit(1)

    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 4)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
    glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
    # glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, gl.GL_TRUE)

    # Create a windowed mode window and its OpenGL context
    global window
    window = glfw.create_window(
        g_camera_3dgs.w, g_camera_3dgs.h, window_name, None, None
    )
    glfw.make_context_current(window)
    glfw.swap_interval(0)
    # glfw.set_input_mode(window, glfw.CURSOR, glfw.CURSOR_NORMAL);
    if not window:
        glfw.terminate()
        print("Could not initialize Window")
        exit(1)

    return window

def cursor_pos_callback(window, xpos, ypos):
    if imgui.get_io().want_capture_mouse:
        g_camera_3dgs.is_leftmouse_pressed = False
        g_camera_3dgs.is_rightmouse_pressed = False
        g_camera_nerf.is_leftmouse_pressed = False
        g_camera_nerf.is_rightmouse_pressed = False
    g_camera_3dgs.process_mouse(xpos, ypos)
    g_camera_nerf.process_mouse(xpos, ypos)

def mouse_button_callback(window, button, action, mod):
    if imgui.get_io().want_capture_mouse:
        return
    pressed = action == glfw.PRESS

    g_camera_3dgs.is_leftmouse_pressed = (button == glfw.MOUSE_BUTTON_LEFT and pressed and update_3dgs_camera)
    g_camera_3dgs.is_rightmouse_pressed = (button == glfw.MOUSE_BUTTON_RIGHT and pressed and update_3dgs_camera)
    g_camera_nerf.is_leftmouse_pressed = (button == glfw.MOUSE_BUTTON_LEFT and pressed and update_nerf_camera)
    g_camera_nerf.is_rightmouse_pressed = (button == glfw.MOUSE_BUTTON_RIGHT and pressed and update_nerf_camera)

def wheel_callback(window, dx, dy):
    if update_3dgs_camera:
        g_camera_3dgs.process_wheel(dx, dy)
    if update_nerf_camera:
        g_camera_nerf.process_wheel(dx, dy)

def key_callback(window, key, scancode, action, mods):
    if action == glfw.REPEAT or action == glfw.PRESS:
        if key == glfw.KEY_Q:
            if update_3dgs_camera:
                g_camera_3dgs.process_roll_key(1)
            if update_nerf_camera:
                g_camera_nerf.process_roll_key(1)
        elif key == glfw.KEY_E:
            if update_nerf_camera:
                g_camera_nerf.process_roll_key(-1)
            if update_3dgs_camera:
                g_camera_3dgs.process_roll_key(-1)

def update_camera_pose_lazy():
    if g_camera_nerf.is_pose_dirty and Visualiser.is_model_loaded():
        Visualiser.cancel_prev_render.set()
        if Visualiser.render_thread is not None:
            Visualiser.render_thread .join()

        Visualiser.cancel_prev_render.clear()

        nerf_gen = Visualiser.create_nerf_renderer(g_camera_nerf.get_transformation_matrix())
        Visualiser.render_thread = threading.Thread(target=Visualiser.run_nerf_renderer, args=(nerf_gen, ), daemon=True)
        Visualiser.render_thread.start()
        g_camera_nerf.is_pose_dirty = False

    if g_camera_3dgs.is_pose_dirty:
        g_renderer.update_camera_pose(g_camera_3dgs)
        g_camera_3dgs.is_pose_dirty = False

def update_camera_intrin_lazy():
    if g_camera_3dgs.is_intrin_dirty:
        g_renderer.update_camera_intrin(g_camera_3dgs)
        g_camera_3dgs.is_intrin_dirty = False

def update_activated_renderer_state(gaus: util_gau.GaussianData):
    g_renderer.update_gaussian_data(gaus)
    g_renderer.sort_and_update(g_camera_3dgs)
    g_renderer.set_scale_modifier(g_scale_modifier)
    g_renderer.set_render_mod(g_render_mode - 3)
    g_renderer.update_camera_pose(g_camera_3dgs)
    g_renderer.update_camera_intrin(g_camera_3dgs)
    g_renderer.set_render_reso(g_camera_3dgs.w, g_camera_3dgs.h)

def window_resize_callback(window, width, height):
    gl.glViewport(0, 0, width, height)
    
    g_camera_3dgs.update_resolution(height, width)
    g_renderer.set_render_reso(width, height)


def update_camera_poses(pose, camera):
    camera.position = pose[:3,3]
    camera.target = util.Camera.extract_target(pose)
    camera.is_pose_dirty = True

def main():
    global g_camera_3dgs, g_camera_nerf, g_renderer, g_renderer_list, g_renderer_idx, g_scale_modifier, g_auto_sort, \
        g_show_control_win, g_show_help_win, g_show_camera_win, \
        g_render_mode, g_render_mode_tables, update_3dgs_camera, update_nerf_camera
        
    imgui.create_context()
    if args.hidpi:
        imgui.get_io().font_global_scale = 1.5
    window = impl_glfw_init()
    impl = GlfwRenderer(window)
    root = tk.Tk()  # used for file dialog
    root.withdraw()

    g_image_renderer = SimpleImageRenderer()
    g_seg_image_renderer = SimpleImageRenderer()
    g_3dgs_seg_image_renderer = SimpleImageRenderer()

    
    glfw.set_cursor_pos_callback(window, cursor_pos_callback)
    glfw.set_mouse_button_callback(window, mouse_button_callback)
    glfw.set_scroll_callback(window, wheel_callback)
    glfw.set_key_callback(window, key_callback)
    
    glfw.set_window_size_callback(window, window_resize_callback)

    # init renderer
    g_renderer_list[BACKEND_OGL] = OpenGLRenderer(g_camera_3dgs.w, g_camera_3dgs.h)
    try:
        from renderer_cuda import CUDARenderer
        g_renderer_list += [CUDARenderer(g_camera_3dgs.w, g_camera_3dgs.h)]
    except ImportError:
        g_renderer_idx = BACKEND_OGL
    else:
        g_renderer_idx = BACKEND_CUDA

    g_renderer = g_renderer_list[g_renderer_idx]

    # gaussian data
    gaussians = util_gau.naive_gaussian()
    update_activated_renderer_state(gaussians)

    i = 0
    # Initialize an RGB array (H, W, 3)
    rgb_array = np.zeros((500, 500, 3), dtype=np.uint8)

    nerf_ind = 0
    loaded_data = None

    error_message = ""
    
    # settings
    while not glfw.window_should_close(window):
        glfw.poll_events()
        impl.process_inputs()
        imgui.new_frame()
        
        gl.glClearColor(0, 0, 0, 1.0)
        gl.glClear(gl.GL_COLOR_BUFFER_BIT)

        update_camera_pose_lazy()
        update_camera_intrin_lazy()
        
        # g_renderer.draw()
        # Bottom Left viewport
        gl.glViewport(0, 0, g_camera_3dgs.w//2, g_camera_3dgs.h//2)
        g_renderer.draw()            

        gl.glViewport(0, g_camera_3dgs.h//2,  g_camera_3dgs.w//2, g_camera_3dgs.h//2)

        i += 1
        if i % 100000 == 0:
            print("RENDERING")
            dominant_indices = g_renderer.get_dominant_gaussian()
            rgb_array = np.zeros((500, 500, 3), dtype=np.uint8)
            # print(dominant_indices.shape)

            color_map = [
                [50, 50, 50],      # Red for class 0
                [0, 255, 0],      # Green for class 1
                [255, 0, 255],      # Blue for class 2
                [255, 255, 0],    # Yellow for class 3
                [0, 255, 255],    # Cyan for class 4
            ]

            # Map class indices to corresponding colors
            for i in range(5):  # Iterate over the 5 possible classes
                rgb_array[dominant_indices%5 == (i-1)] = color_map[i]
            # print(rgb_array)
            # print(np.unique(rgb_array.reshape(-1, 3), axis=0))

        g_3dgs_seg_image_renderer.update_texture(rgb_array)
        g_3dgs_seg_image_renderer.draw()

        # Bottom Right viewport
        gl.glViewport(g_camera_3dgs.w//2, 0, g_camera_3dgs.w//2, g_camera_3dgs.h//2)

        if Visualiser.is_model_loaded():
            image_data, seg_image_data = Visualiser.get_nerf_image()  # Your function that returns the image
            g_image_renderer.update_texture(image_data)
            g_image_renderer.draw()

            # Top Right viewport
            gl.glViewport(g_camera_3dgs.w//2, g_camera_3dgs.h//2, g_camera_3dgs.w//2, g_camera_3dgs.h//2)
            g_seg_image_renderer.update_texture(seg_image_data)
            g_seg_image_renderer.draw()
        
        # imgui ui
        if imgui.begin_main_menu_bar():
            if imgui.begin_menu("Window", True):
                clicked, g_show_control_win = imgui.menu_item(
                    "Show Control", None, g_show_control_win
                )
                clicked, g_show_help_win = imgui.menu_item(
                    "Show Help", None, g_show_help_win
                )
                clicked, g_show_camera_win = imgui.menu_item(
                    "Show Camera Control", None, g_show_camera_win
                )
                imgui.end_menu()
            imgui.end_main_menu_bar()


        
        if g_show_control_win:
            if imgui.begin("Control", True):
                imgui.text(f"fps = {imgui.get_io().framerate:.1f}")


                imgui.text(f"# of Gaus = {len(gaussians)}")
                if imgui.button(label='Load 3DGS'):
                    file_path = filedialog.askopenfilename(title="open ply",
                        initialdir="C:\\Users\\MSI_NB\\Downloads\\viewers",
                        filetypes=[('ply file', '.ply')]
                        )
                    if file_path:
                        try:
                            gaussians = util_gau.load_ply(file_path)
                            g_renderer.update_gaussian_data(gaussians)
                            g_renderer.sort_and_update(g_camera_3dgs)
                        except RuntimeError as e:
                            print(f"Error loading 3DGS: {e}")
                            pass
                
                if imgui.button(label='Load NeRF'):
                    file_path = filedialog.askopenfilename(title="open nerf model",
                        initialdir="C:\\Users\\MSI_NB\\Downloads\\viewers",
                        filetypes=[('Nerf model', '.pth')]
                        )
                    if file_path:
                        try:
                            # Visualiser.load_model(file_path)
                            Visualiser(file_path, num_of_labels=4, focal=focal)
                            update_camera_pose_lazy()
                            g_camera_nerf.is_pose_dirty = True
                        except RuntimeError as e:
                            print(f"Error loading NeRF: {e}")
                            pass
                
                if imgui.button(label='Load camera poses'):
                    file_path = filedialog.askopenfilename(title="open .npz file",
                        initialdir="C:\\Users\\MSI_NB\\Downloads\\viewers",
                        filetypes=[('Camera data', '.npz')]
                        )
                    if file_path:
                        try:
                            # Set the camera pos and target pose
                            loaded_data = torch.from_numpy(np.load(file_path)['poses_train']).numpy()
                            update_camera_poses(loaded_data[nerf_ind], g_camera_nerf)
                            update_camera_poses(loaded_data[nerf_ind], g_camera_3dgs)
                            update_camera_pose_lazy()
                        except RuntimeError as e:
                            print(f"Error loading the camera poses: {e}")
                            pass
                
                imgui.same_line()

                if imgui.button(label='Next'):
                    if loaded_data is not None:
                        update_camera_poses(loaded_data[nerf_ind], g_camera_nerf)
                        update_camera_poses(loaded_data[nerf_ind], g_camera_3dgs)
                        nerf_ind += 1
                        update_camera_pose_lazy()
                    else:
                        print("No camera poses loaded!")

                changed, update_3dgs_camera = imgui.checkbox("Update 3DGS camera", update_3dgs_camera)           
                changed, update_nerf_camera = imgui.checkbox("Update NeRF camera", update_nerf_camera)

                g_renderer.sort_and_update(g_camera_3dgs)
                
                if imgui.button(label='save image'):
                    width, height = glfw.get_framebuffer_size(window)
                    nrChannels = 3
                    stride = nrChannels * width
                    stride += (4 - stride % 4) if stride % 4 else 0
                    gl.glPixelStorei(gl.GL_PACK_ALIGNMENT, 4)
                    gl.glReadBuffer(gl.GL_FRONT)
                    bufferdata = gl.glReadPixels(0, 0, width//2, height//2, gl.GL_RGB, gl.GL_UNSIGNED_BYTE)
                    img = np.frombuffer(bufferdata, np.uint8, -1).reshape(height//2, width//2, 3)
                    imageio.imwrite("save.png", img[::-1])


                    dominant_indices = g_renderer.get_dominant_gaussian()
                    print(f"Saving img {img.shape} and gaus inds {dominant_indices.shape}")
                    img_gaus_inds_data = ([], [])
                    if os.path.exists("gaus_ind_data.npz"):
                        loaded_data = np.load("gaus_ind_data.npz")
                        img_gaus_inds_data = (loaded_data["images"], loaded_data["gaus_inds"])
                        img_gaus_inds_data = (np.vstack((img_gaus_inds_data[0], np.expand_dims(img, axis=0))), np.vstack((img_gaus_inds_data[1], np.expand_dims(dominant_indices, axis=0))))
                    else:
                        img_gaus_inds_data = (np.array([img]), np.array([dominant_indices]))
                
                    np.savez("gaus_ind_data.npz", images=img_gaus_inds_data[0], gaus_inds=img_gaus_inds_data[1])

                imgui.end()

        if g_show_help_win:
            imgui.begin("Help", True)
            imgui.text("Open Gaussian Splatting PLY file \n  by click 'Load ply' button")
            imgui.text("Open NeRF PTH file \n  by click 'Load NeRF' button")
            imgui.text("Open Camera pose npz file \n  by click 'Load camera pose' button")
            imgui.text("Use left click & move to rotate camera")
            imgui.text("Use right click & move to translate camera")
            imgui.text("Press Q/E to roll camera")
            imgui.text("Use scroll to zoom in/out")
            # imgui.text("Use control panel to change setting")
            imgui.text("You can also freeze one of the cameras by not updating the camera to align the views")
            imgui.end()
        
        imgui.render()
        impl.render(imgui.get_draw_data())
        glfw.swap_buffers(window)

    impl.shutdown()
    glfw.terminate()


if __name__ == "__main__":
    global args
    parser = argparse.ArgumentParser(description="NeUVF editor with optional HiDPI support.")
    parser.add_argument("--hidpi", action="store_true", help="Enable HiDPI scaling for the interface.")
    args = parser.parse_args()


    main()
