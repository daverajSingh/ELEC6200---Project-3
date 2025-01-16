import argparse
import shutil
import sys
import config
import os 

"""
Pipeline to fully generate NeRF and 3DGS model from a video.
"""

NORMAL_RES = config.NORMAL_RES
UPSCALED_RES_N = config.UPSCALED_RES_N
DOWNSCALED_RES_N = config.DOWNSCALED_RES_N
UPSCALED_RES = (int(NORMAL_RES[0]/UPSCALED_RES_N), int(NORMAL_RES[1]/UPSCALED_RES_N))
DOWNSCALED_RES = (int(NORMAL_RES[0]/DOWNSCALED_RES_N), int(NORMAL_RES[1]/DOWNSCALED_RES_N))

def set_up_colmap_folders():
    os.makedirs(config.COLMAP_OUPUT_PATH, exist_ok=True)
    TRANSFORMS_PATH = os.path.join(config.COLMAP_OUPUT_PATH,"transforms.json")
    TEXT_PATH = os.path.join(config.COLMAP_OUPUT_PATH,"colmap_text")
    os.makedirs(TEXT_PATH, exist_ok=True)
    return TRANSFORMS_PATH, TEXT_PATH

def extract_frames_from_video():
    from utils.extract_frames import extract_frames
    print("Extracting frames using extract_frames.py")

    extract_frames(config.VIDEO_PATH, config.IMAGE_PATH, NORMAL_RES)

def get_cubemap_projection():
    from utils.cubemap_leftnright import process_equirectangular_images

    print("Applying cubemap projection to extracted frames")

    process_equirectangular_images(config.IMAGE_PATH, os.join(config.IMAGE_PATH, "cubemap"))
    for item in os.listdir(config.IMAGE_PATH):
        item_path = os.path.join(config.IMAGE_PATH, item)
        if os.path.isfile(item_path):
            os.remove(item_path)
    
    for item in os.listdir(os.join(config.IMAGE_PATH, "cubemap")):
            source_item = os.path.join(os.join(config.IMAGE_PATH, "cubemap"), item)
            dest_item = os.path.join(os.join(config.IMAGE_PATH, "cubemap"), item)
            shutil.move(source_item, dest_item)

def get_camera_pose_from_images_nerf():
    from utils.extract_poses import main as extract_poses
    TRANSFORMS_PATH, TEXT_PATH = set_up_colmap_folders()

    print("Extracting image camera poses using extract_poses.py")
    extract_poses(
        db_path=os.path.join(config.COLMAP_OUPUT_PATH, "colmap.db"),
        img_path= config.IMAGE_PATH,
        text_path = TEXT_PATH,
        output_path = TRANSFORMS_PATH,
        colmap_camera_model="SIMPLE_RADIAL",
        colmap_matcher="exhaustive",
        aabb_scale=8
    )

def get_camera_pose_from_images_nerf_from_3dgs():
    from utils.extract_poses import main as extract_poses

    TRANSFORMS_PATH, TEXT_PATH = set_up_colmap_folders()

    print("Extracting camera poses using 3DGS poses using extract_poses.py")
    extract_poses(
        img_path= config.IMAGE_PATH,
        text_path = TEXT_PATH,
        output_path = TRANSFORMS_PATH,
        dgs_path= os.path.join(config.SCENE_PATH, "sparse"),
    )

def get_camera_pose_from_images_3dgs():
    # os.rename(config.COLMAP_OUPUT_PATH, os.join(config.SCENE_PATH, "input"))
    print("Getting image camera poses using 3dgs convert.py")

    err = os.system(f"python 3dgs/convert.py -s {config.SCENE_PATH}")
    if err:
        print("FATAL: command failed")
        sys.exit(err)

    # _, TEXT_PATH = set_up_colmap_folders()
    # shutil.copy(os.path.join(config.SCENE_PATH, "distorted/sparse/0/text/cameras.txt") , TEXT_PATH)
    # shutil.copy(os.path.join(config.SCENE_PATH, "distorted/sparse/0/transforms.json") , config.COLMAP_OUPUT_PATH)

def get_semantic_labels_from_images():
    from segmentation.detector import Detector

    print("Extracting semantic outputs from frames")
    detector = Detector()
    detector.classifyFrames(config.IMAGE_PATH , config.SEG_IMAGE_PATH)


def run_3dgs():
    err = os.system(f"python 3dgs/train.py -s {config.SCENE_PATH} --disable_viewer")
    if err:
        print("FATAL: command failed")
        sys.exit(err)

def format_data_in_npz_format(extract_from_video=True):
    from utils.extract_frames import extract_frames
    from utils.format_data import main as format_data

    TRANSFORMS_PATH, TEXT_PATH = set_up_colmap_folders()

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

def run_instant_nerf():
    from nerf.instant_nerf import main as run_nerf
    print("Running instant NeRF")
    run_nerf(config.DATA_OUTPUT_PATH)

def parse_args():
	parser = argparse.ArgumentParser(description="Pipeline to turn a video into NeRF")

	parser.add_argument("--full_pipeline", action="store_true", help="Runs end-end pipeline to get NeRF and 3DGS models")
	parser.add_argument("--gs_pipeline", action="store_true", help="Runs end-end pipeline to get 3DGS model")
	parser.add_argument("--nerf_pipeline", action="store_true", help="Runs end-end pipeline to get NeRF model")


	parser.add_argument("--extract_frames", action="store_true", help="Extracts frames from video")
	parser.add_argument("--apply_cubemap_projection", action="store_true", help="Applies cubemap projection to extracted frames")
	parser.add_argument("--get_semantic_labels", action="store_true", help="Extracts semantic outputs from frames")
	parser.add_argument("--get_camera_pose", action="store_true", help="Runs colmap to extract camera poses from images")
	parser.add_argument("--run_3dgs", action="store_true", help="Trains 3D gaussian splatting model")
	parser.add_argument("--same_camera_pose", action="store_true", help="Uses the same camera poses for both 3dgs and NeRF")
	parser.add_argument("--format_data", action="store_true", help="Formats data to be used by the nerf algorithm")
	parser.add_argument("--from_video", action="store_true", help="Extracts frames from video")
	parser.add_argument("--run_nerf", action="store_true", help="Runs fast nerf")
	args = parser.parse_args()
	return args


if __name__=='__main__':
    args = parse_args()

    if args.full_pipeline:
        extract_frames_from_video()
        if args.apply_cubemap_projection:
            get_cubemap_projection()
        get_semantic_labels_from_images()
        get_camera_pose_from_images_3dgs()
        if args.same_camera_pose:
            get_camera_pose_from_images_nerf_from_3dgs()
        else:
            get_camera_pose_from_images_nerf()
        run_3dgs()
        format_data_in_npz_format(extract_from_video=True)
        run_instant_nerf()
    elif args.gs_pipeline:
        extract_frames_from_video()
        if args.apply_cubemap_projection:
            get_cubemap_projection()
        get_camera_pose_from_images_3dgs()
        run_3dgs()
    elif args.nerf_pipeline:
        extract_frames_from_video()
        if args.apply_cubemap_projection:
            get_cubemap_projection()
        get_semantic_labels_from_images()
        get_camera_pose_from_images_nerf()
        format_data_in_npz_format(extract_from_video=True)
        run_instant_nerf()
    else:
        if args.extract_frames:
            extract_frames_from_video()
        if args.apply_cubemap_projection:
            get_cubemap_projection()
        if args.get_semantic_labels:
            get_semantic_labels_from_images()
        if args.get_camera_pose:
            if args.run_3dgs:
                get_camera_pose_from_images_3dgs()
            if args.same_camera_pose:
                get_camera_pose_from_images_nerf_from_3dgs()
            else:
                get_camera_pose_from_images_nerf()
        if args.format_data:
            format_data_in_npz_format(extract_from_video=args.from_video)
        if args.run_nerf:
            run_instant_nerf()
        if args.run_3dgs:
            run_3dgs()