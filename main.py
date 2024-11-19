from detector import *

detector = Detector()

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
            frame = cv2.resize(frame, SAVE_RES, interpolation=cv2.INTER_AREA)
            # Save the frame
            cv2.imwrite(output_filename, frame)
            
            # Increment frame counter
            frame_count += 1
        
        # Increment current frame
        current_frame += 1
    
    # Release the video capture object
    video.release()
    
    print(f"Extracted {frame_count - 1} frames to {output_folder}")

def convert_images_to_video(input_folder, output_file, fps):
    # Get the list of image files in the input folder
    image_files = sorted([f for f in os.listdir(input_folder) if f.endswith('.jpg') or f.endswith('.png')])

    # Read the first image to get its dimensions
    first_image = cv2.imread(os.path.join(input_folder, image_files[0]))
    height, width, _ = first_image.shape

    # Create a VideoWriter object to save the video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Specify the codec for the output video file
    video = cv2.VideoWriter(output_file, fourcc, fps, (width, height))

    # Iterate over each image and write it to the video
    for image_file in image_files:
        image_path = os.path.join(input_folder, image_file)
        frame = cv2.imread(image_path)
        video.write(frame)

    # Release the video writer and close the video file
    video.release()
    cv2.destroyAllWindows()

# Video 1

#extract_frames('IMG_2471.mov', 'Video3', (1080, 1920), 30)

detector.classifyFrames("Video3", "Result3")

convert_images_to_video("Result3", "Video3.mp4", 30)


