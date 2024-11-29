from detector import *
from helper import *
import ffmpeg

detector = Detector()

def main():
    ffmpeg.convert_to_equirectangular("R0010072[1].mp4", "image360", 5, 1)


if __name__ == "__main__":
    main()