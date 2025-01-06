import subprocess
import os

def convert_to_equirectangular(input_video, output_folder, fps, quality):
    """
    Extracts Frames from 360 Video to Equirectangular Images
    
    MUST HAVE FFMPEG INSTALLED FOR THIS TO WORK
    
    Parameters:
    - input_video: Path to the input video file
    - output_folder: Directory where frames will be saved
    - fps: Frames per second to extract
    - quality: Quality of the images extracted
    """
    
    ffmpeg_command = [
        "ffmpeg",
        "-i", input_video,
        "-vf", f"fps={fps}",
        "-qscale:v", str(quality),
        output_folder+"/image_%04d.jpg"
    ]      
    
    os.makedirs(output_folder, exist_ok=True)
    for filename in os.listdir(output_folder):
        file_path = os.path.join(output_folder, filename)
        if os.path.isfile(file_path):
            os.unlink(file_path)
            
    try:
        subprocess.run(ffmpeg_command, check=True)
        print("Images extracted successfully.")
    except subprocess.CalledProcessError as e:
        print("An error occurred:", e)


