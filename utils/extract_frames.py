import cv2
import os

def extract_frames(video_path, output_folder, SAVE_RES, fps=3):
    """
    Extract frames from a video at specified frames per second.
    
    Parameters:
    - video_path: Path to the input video file
    - output_folder: Directory where frames will be saved
    - fps: Frames per second to extract (default 30)
    """
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    for filename in os.listdir(output_folder):
        file_path = os.path.join(output_folder, filename)
        if os.path.isfile(file_path):
            os.unlink(file_path)
    
    # Open the video file
    video = cv2.VideoCapture(video_path)
    
    # Get the video's original frames per second
    original_fps = video.get(cv2.CAP_PROP_FPS)
    
    # Calculate frame skip to achieve desired fps
    frame_skip = int(original_fps / fps)
    
    # Counter for naming frames
    frame_count = 1
    
    # Current frame to read
    current_frame = 0
    
    while True:
        # Read a frame from the video
        ret, frame = video.read()
        
        # Break the loop if no more frames
        if not ret:
            break
        
        # Extract frame at specified interval
        if current_frame % frame_skip == 0:
            # Construct output filename
            output_filename = os.path.join(output_folder, f"{frame_count:05d}.jpg")
            # frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

            frame = cv2.resize(frame, SAVE_RES, interpolation=cv2.INTER_LANCZOS4) # INTER_AREA
            # frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
            # Save the frame
            cv2.imwrite(output_filename, frame)
            
            # Increment frame counter
            frame_count += 1
        
        # Increment current frame
        current_frame += 1
    
    # Release the video capture object
    video.release()
    
    print(f"Extracted {frame_count - 1} frames to {output_folder}")


# SAVE_RES = (48, 27)
# Example usage
if __name__ == "__main__":
    VIDEO_PATH = "video/ChairVideo.MOV"
    OUTPUT_PATH = "images"
    # SAVE_RES = (1920, 1080)
    SAVE_RES = (192, 108)
    # Extract frames
    extract_frames(VIDEO_PATH, OUTPUT_PATH, SAVE_RES)