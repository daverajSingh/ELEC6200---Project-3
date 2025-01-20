# config.py
# VIDEO_PATH = "video/can_video.mp4"
# SEG_VIDEO_PATH = "video/can_segment.mp4"
SCENE_PATH = "bottlenmouse2"
IMAGE_PATH = f"{SCENE_PATH}/images"
SEG_IMAGE_PATH = f"{SCENE_PATH}/images_segmented"
NORMAL_RES = (512, 512)
# NORMAL_RES = (480, 1920)
UPSCALED_RES_N = 1
DOWNSCALED_RES_N = 4
DATA_OUTPUT_PATH = "nerf_formated_data_small.npz"
COLMAP_OUPUT_PATH = f'{SCENE_PATH}\\colmap_output'
