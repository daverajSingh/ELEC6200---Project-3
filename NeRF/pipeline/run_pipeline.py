import argparse
import config
import os 

"""
Pipeline to fully generate NeRF model from a video.
"""

os.makedirs(config.COLMAP_OUPUT_PATH, exist_ok=True)
TRANSFORMS_PATH = os.path.join(config.COLMAP_OUPUT_PATH,"transforms.json")
TEXT_PATH = os.path.join(config.COLMAP_OUPUT_PATH,"colmap_text")
NORMAL_RES = config.NORMAL_RES
UPSCALED_RES_N = config.UPSCALED_RES_N
DOWNSCALED_RES_N = config.DOWNSCALED_RES_N
UPSCALED_RES = (int(NORMAL_RES[0]/UPSCALED_RES_N), int(NORMAL_RES[1]/UPSCALED_RES_N))
DOWNSCALED_RES = (int(NORMAL_RES[0]/DOWNSCALED_RES_N), int(NORMAL_RES[1]/DOWNSCALED_RES_N))

def get_camera_pose_from_video():
    from extract_frames import extract_frames
    from extract_poses import main as extract_poses

    print("Extracting upscaled frames using extract_frames.py")
    extract_frames(config.VIDEO_PATH, config.IMAGE_PATH, UPSCALED_RES)

    print("Extracting camera poses using extract_poses.py")
    extract_poses(
        db_path=os.path.join(config.COLMAP_OUPUT_PATH, "colmap.db"),
        img_path= config.IMAGE_PATH,
        text_path = TEXT_PATH,
        output_path = TRANSFORMS_PATH,
        aabb_scale=1
    )

def get_camera_pose_from_images():
    from extract_poses import main as extract_poses

    print("Extracting image camera poses using extract_poses.py")
    extract_poses(
        db_path=os.path.join(config.COLMAP_OUPUT_PATH, "colmap.db"),
        img_path= config.IMAGE_PATH,
        text_path = TEXT_PATH,
        output_path = TRANSFORMS_PATH,
        colmap_camera_model="SIMPLE_RADIAL",
        colmap_matcher="exhaustive",
        aabb_scale=1
    )

def format_data_in_npz_format(extract_from_video=True):
    from extract_frames import extract_frames
    from format_data import main as format_data

    if extract_from_video:
        print("Extracting downscaled frames using extract_frames.py")
        extract_frames(config.VIDEO_PATH, config.IMAGE_PATH, DOWNSCALED_RES)
        extract_frames(config.SEG_VIDEO_PATH, config.SEG_IMAGE_PATH, DOWNSCALED_RES)
    print("Formating data to npz np format using format_data.py")
    format_data(
        config.IMAGE_PATH,
        config.SEG_IMAGE_PATH,
        TRANSFORMS_PATH,
        os.path.join(TEXT_PATH, "cameras.txt"),
        output_path=config.DATA_OUTPUT_PATH
    )

def run_nerf_alg():
    from nerf import main as run_nerf
    print("Running NeRF")
    run_nerf(config.DATA_OUTPUT_PATH)

def run_instant_nerf():
    from instant_nerf import main as run_nerf
    print("Running instant NeRF")
    run_nerf(config.DATA_OUTPUT_PATH)

def parse_args():
	parser = argparse.ArgumentParser(description="Pipeline to turn a video into NeRF")

	parser.add_argument("--get_camera_pose", action="store_true", help="Extracts frames and runs colmap to extract camera poses")
	parser.add_argument("--get_camera_pose_from_images", action="store_true", help="Runs colmap to extract camera poses from images")
	parser.add_argument("--format_data", action="store_true", help="Formats data to be used by the nerf algorithm")
	parser.add_argument("--from_video", action="store_true", help="Extracts frames from video")
	parser.add_argument("--run_nerf", action="store_true", help="Runs nerf")
	parser.add_argument("--run_instant_nerf", action="store_true", help="Runs fast nerf (Recommended)")
	args = parser.parse_args()
	return args


if __name__=='__main__':
    args = parse_args()
    if args.get_camera_pose:
        get_camera_pose_from_video()
    if args.get_camera_pose_from_images:
        get_camera_pose_from_images()
    if args.format_data:
        format_data_in_npz_format(extract_from_video=args.from_video)
    if args.run_nerf:
        run_nerf_alg()
    if args.run_instant_nerf:
        run_instant_nerf()