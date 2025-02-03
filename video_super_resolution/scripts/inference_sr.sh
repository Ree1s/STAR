#!/bin/bash
# Set HIP_VISIBLE_DEVICES to specify which GPU(s) to use
export HIP_VISIBLE_DEVICES=0

# Set MIOpen cache and database paths to avoid SQLite errors
# export MIOPEN_USER_DB_PATH="/tmp/my-miopen-cache"
export MIOPEN_USER_DB_PATH="/group/ossdphi_algo_scratch_14/sichegao/miopen_cache"

export MIOPEN_CUSTOM_CACHE_DIR=${MIOPEN_USER_DB_PATH}

# Clear and recreate the MIOpen cache directory
rm -rf ${MIOPEN_USER_DB_PATH}
mkdir -p ${MIOPEN_USER_DB_PATH}
# Folder paths
video_folder_path='./input/video/'
txt_file_path='./input/text/prompt.txt'

# Get all .mp4 files in the folder using find to handle special characters
mapfile -t mp4_files < <(find "$video_folder_path" -type f -name "*.mp4")

# Print the list of MP4 files
echo "MP4 files to be processed:"
for mp4_file in "${mp4_files[@]}"; do
    echo "$mp4_file"
done

# Read lines from the text file, skipping empty lines
mapfile -t lines < <(grep -v '^\s*$' "$txt_file_path")

# The number of video frames processed simultaneously during each denoising process.
frame_length=24

# Debugging output
echo "Number of MP4 files: ${#mp4_files[@]}"
echo "Number of lines in the text file: ${#lines[@]}"

# Ensure the number of video files matches the number of lines
if [ ${#mp4_files[@]} -ne ${#lines[@]} ]; then
    echo "Number of MP4 files and lines in the text file do not match."
    exit 1
fi

# Loop through video files and corresponding lines
for i in "${!mp4_files[@]}"; do
    mp4_file="${mp4_files[$i]}"
    line="${lines[$i]}"
    
    # Extract the filename without the extension
    file_name=$(basename "$mp4_file" .mp4)
    
    echo "Processing video file: $mp4_file with prompt: $line"
        
    # Run Python script with parameters
    python \
        ./video_super_resolution/scripts/inference_sr.py \
        --solver_mode 'fast' \
        --steps 15 \
        --input_path "${mp4_file}" \
        --model_path ./pretrained_weight/heavy_deg.pt \
        --prompt "${line}" \
        --upscale 4 \
        --max_chunk_len ${frame_length} \
        --file_name "${file_name}.mp4" \
        --save_dir ./apt_results_i2vheavy_wprompt
done

echo "All videos processed successfully."
