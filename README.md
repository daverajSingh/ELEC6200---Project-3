# NeRF and 3DGS pipeline with visualiser
This project contains a pipeline to convert a video into NeRF (Neural Radiance Fields) and 3D Gaussian Splatting (3DGS) models.

It also contains a custom built visualiser to view outputs of both models simultaneously along with their semantic outputs.

## Set up

To run 3D gaussian splatting please follow set up on the main repo: [gaussian-splatting-Windows](https://github.com/jonstephens85/gaussian-splatting-Windows)

Set up a conda environment: `conda env create -f environment.yml` \
Activate environment: `conda activate pipeline_env`


## Running The Pipeline

![Pipeline diagram](https://github.com/user-attachments/assets/20d97209-312b-43a3-a91e-af7302198754)*Pipeline diagram*

### Video Demonstration of pipeline
[![Video Demo](https://img.youtube.com/vi/-2g7QiP8JlI/0.jpg)](https://www.youtube.com/watch?v=-2g7QiP8JlI)

### Arguments

`--full_pipeline`: Runs the entire pipeline to generate both NeRF and 3DGS models. \
`--gs_pipeline`: Runs the pipeline to generate the 3DGS model.\
`--nerf_pipeline`: Runs the pipeline to generate the NeRF model.\
`--extract_frames`: Extracts frames from the input video.\
`--apply_cubemap_projection`: Applies cubemap projection to the extracted frames.\
`--get_semantic_labels`: Extracts semantic labels from the frames.\
`--get_camera_pose`: Runs COLMAP to extract camera poses from the images.\
`--run_3dgs`: Trains the 3D Gaussian Splatting model.\
`--format_data`: Formats the data to be used by the NeRF algorithm.\
`--from_video`: Extracts frames from the video.\
`--run_nerf`: Runs the fast NeRF algorithm.

Run with an additional `--360-video` argument if 360 video is being used.
### Usage
To run the script with a specific argument, use the following command:
`python run_pipeline.py [argument]`

For example, to run the entire pipeline:
`python run_pipeline.py --full_pipeline`\
`python run_pipeline.py --gs_pipeline` to run only 3DGS pipeline.\
`python run_pipeline.py --nerf_pipeline` to run only NeRF pipeline.

> Due to the modular approach it is possible to just run parts of this pipeline.

`python run_pipeline.py --extract-frames`: To extract video into frames. \
`python run_pipeline.py --get_semantic_labels`: Runs image segmentation model on the extracted images.

Please refer to all the arguments in `run_pipeline.py` to see all the settings and customisations.

## Visualiser

`python visualiser/main.py`: To open the visualiser to view 3dgs and nerf outputs side by side. Will have to load the models once the visualiser is open.

The visualiser has been adapted from [GaussianSplattingViewer](https://github.com/limacv/GaussianSplattingViewer/) to also support NeRFs.

Use the `load 3dgs model` and navigate to .ply file to open 3DGS model  \
Use the `load nerf model` and navigate to .pth weights file to open NeRF model \
Use the `load camera poses` and navigate to .npz file to open the orignal camera pose location. You can then click the next camera location to iterate through the camera poses. Or simply move around by using right and left mouse buttons and dragging.

Bottom Right: NeRF output \
Bottom Left: 3DGS output \
Top Right: Nerf Segmentation output

![Visualiser](https://github.com/user-attachments/assets/d0a4c24f-dd22-450e-a8c3-155677f427da)

