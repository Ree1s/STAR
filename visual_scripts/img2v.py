import os
import cv2
import numpy as np
import imageio

def generate_videos_from_udm10(dataset_path, output_path, fps=24):
    """
    Generate MP4 files from UDM10 dataset structure without altering frame dimensions.

    :param dataset_path: Path to the UDM10 dataset (e.g., 'UDM10').
    :param output_path: Path to save the generated MP4 files.
    :param fps: Frames per second for the videos.
    """
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # Iterate through the folders representing videos
    for category in os.listdir(dataset_path):
        category_path = os.path.join(dataset_path, category)
        if not os.path.isdir(category_path):
            continue

        for blur_level in os.listdir(category_path):
            blur_path = os.path.join(category_path, blur_level)
            if not os.path.isdir(blur_path):
                continue

            # Collect frames for each video
            frames = []
            for frame_file in sorted(os.listdir(blur_path)):
                if frame_file.endswith('.png'):
                    frame_path = os.path.join(blur_path, frame_file)
                    frame = cv2.imread(frame_path)
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
                    frames.append(frame)

            if not frames:
                print(f"No frames found in {blur_path}, skipping.")
                continue

            # Generate MP4 video
            video_name = f"{category}_{blur_level}.mp4"
            video_path = os.path.join(output_path, video_name)

            try:
                imageio.mimwrite(video_path, np.array(frames), fps=fps, codec="libx264", macro_block_size=1)
                print(f"Video saved: {video_path}")
            except Exception as e:
                print(f"Error creating video for {blur_path}: {e}")




if __name__ == "__main__":
    dataset_path = "/group/ossdphi_algo_scratch_14/sichegao/datasets/UDM10"  # Replace with the path to your UDM10 dataset
    output_path = "/group/ossdphi_algo_scratch_14/sichegao/datasets/UDM10_mp4"  # Replace with your desired output directory
    fps = 24  # Adjust frames per second as needed

    generate_videos_from_udm10(dataset_path, output_path, fps)
