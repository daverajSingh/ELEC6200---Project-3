from detector import *
from helper import *
import ffmpeg

def main(video_path):
    detector = Detector()
    detector.classifyFrames("images_cubemap" , "images_segmented")
