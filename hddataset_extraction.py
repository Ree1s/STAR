import os
import shutil
import pandas as pd

# Data directories containing the videos
data_dirs = [
    '/scratch1_nvme_2/workspace/scgao/video',
    '/scratch1_nvme_1/workspace/scgao/video'
]

# Path to the original CSV file
csv_path = '/scratch1_nvme_2/workspace/scgao/data/train/OpenVidHD.csv'

# Destination directories for videos and CSV
destination_video_dir = './dataset/OpenVidHD/video/'
destination_csv_path = './dataset/OpenVidHD/data/train/OpenVidHD_sub.csv'

# Create the destination video directory if it doesn't exist
os.makedirs(destination_video_dir, exist_ok=True)

# Read the CSV file into a DataFrame
df = pd.read_csv(csv_path)

# Filter rows where 'frames' > 32
filtered_df = df[df['frame'] > 32]

# Select up to 1,000 videos
selected_df = filtered_df.head(1000)

# Initialize a counter for copied videos
copied_count = 0

# Iterate over the selected DataFrame rows
for _, row in selected_df.iterrows():
    video_name = row['video']
    # Search for the video in the data directories
    found = False
    for data_dir in data_dirs:
        source_path = os.path.join(data_dir, video_name)
        if os.path.exists(source_path):
            # Copy the video to the destination directory
            shutil.copy(source_path, destination_video_dir)
            copied_count += 1
            print(f'Copied {video_name} to {destination_video_dir}')
            found = True
            break
    if not found:
        print(f'Warning: {video_name} not found in specified directories.')

# Save the selected DataFrame to a new CSV file
selected_df.to_csv(destination_csv_path, index=False)
print(f'Saved metadata of selected videos to {destination_csv_path}')

# Output the total number of copied videos
print(f'Total number of videos copied: {copied_count}')
