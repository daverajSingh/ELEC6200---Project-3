This code is best used when taking a 360 video in a small space where the person taking the video is in frame of the video, which is also why that specific frame should be discarded.

The code projects equirectangular images to cubemap images and it only saves front and back (cubemap_frontnback.py) or left and right (cubemap_leftnright.py).

Cubemap orientation is as such:

            [TOP]
      [LEFT][FRONT][RIGHT][BACK]
            [BOTTOM]

Run code with command line: 
`python cubemap_frontnback.py -i \path\to\equirectangular\images -o \path\to\output\folder`
