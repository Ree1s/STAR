import os
import cv2

def convert_video_to_frames(video_path, output_dir, frame_rate=24):
    """
    Convert a video to individual frames and save them as images.

    Args:
        video_path (str): Path to the video file.
        output_dir (str): Directory to save extracted frames.
        frame_rate (int): Frame rate to extract frames. Default is 24.
    """
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Open the video file
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return

    # Get the video frame rate
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = round(fps / frame_rate)  # Calculate interval to match the desired frame rate

    frame_count = 0
    saved_frame_count = 0
    while True:
        ret, frame = cap.read()
        
        if not ret:
            break  # No more frames to read

        # Extract frame at the given interval
        if frame_count % frame_interval == 0:
            # Create a filename for the frame (e.g., frame_000001.png)
            frame_filename = f"{saved_frame_count:03d}.png"
            frame_path = os.path.join(output_dir, frame_filename)
            # Save the frame as an image
            cv2.imwrite(frame_path, frame)
            saved_frame_count += 1

        frame_count += 1

    cap.release()
    print(f"Video {video_path} converted to {saved_frame_count} frames.")

def convert_videos_in_directory(input_dir, output_dir, frame_rate=24):
    """
    Convert all MP4 videos in a directory to frames.

    Args:
        input_dir (str): Directory containing MP4 video files.
        output_dir (str): Directory where extracted frames will be saved.
        frame_rate (int): Frame rate to extract frames. Default is 24.
    """
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Get all MP4 files in the input directory
    video_files = [f for f in os.listdir(input_dir) if f.endswith('.mp4')]

    # Process each video file
    for video_file in video_files:
        video_path = os.path.join(input_dir, video_file)
        video_output_dir = os.path.join(output_dir, os.path.splitext(video_file)[0])  # Create subfolder for each video
        convert_video_to_frames(video_path, video_output_dir, frame_rate)

# Example usage:
input_directory = 'input/video'  # Directory containing MP4 files
output_directory = 'aigc_frames'  # Directory to save extracted frames

convert_videos_in_directory(input_directory, output_directory, frame_rate=24)
