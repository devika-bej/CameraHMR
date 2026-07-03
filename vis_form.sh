#!/bin/bash

# Check if batch number x is provided
if [ -z "$1" ]; then
    echo "Usage: $0 <x>"
    exit 1
fi

x=$1

image_dir="../NGT_Temporary_${x}"
result_dir="../NGT_Experiments_${x}"

# New centralized video output directory
video_result_dir="../NGT_Video_Results_${x}"

# Create video result directory
mkdir -p "$video_result_dir"

# Ensure result directory exists
if [ ! -d "$result_dir" ]; then
    echo "Error: Directory $result_dir not found."
    exit 1
fi

# Loop through sample directories
for dir_path in "$result_dir"/sample*/; do

    [ -d "$dir_path" ] || continue

    samplei=$(basename "$dir_path")

    # Count files only
    file_count=$(find "$dir_path" -maxdepth 1 -type f | wc -l)

    if [ "$file_count" -eq 10 ]; then

        echo "Processing $samplei..."

        ############################################
        # Block 1: Final Estimate
        ############################################

        temp_dir="$result_dir/$samplei/img_temp_final"

        python3 -W ignore dataset_vis_smplx_gemini.py \
            "$image_dir/$samplei/input_front" \
            "$result_dir/$samplei/final_estimate.npz" \
            "$temp_dir"

        python3 ~/image_to_video.py \
            "$temp_dir" \
            "$video_result_dir/${samplei}_optimized.mp4"

        rm -rf "$temp_dir"

        ############################################
        # Block 2: CamHMR Front
        ############################################

        temp_dir="$result_dir/$samplei/img_temp_camhmr"

        python3 -W ignore dataset_vis_smplx_gemini.py \
            "$image_dir/$samplei/input_front" \
            "$result_dir/$samplei/camhmr_front.npz" \
            "$temp_dir"

        python3 ~/image_to_video.py \
            "$temp_dir" \
            "$video_result_dir/${samplei}_camhmr.mp4"

        rm -rf "$temp_dir"

        ############################################
        # Block 3: Hand Optimization Front
        ############################################

        temp_dir="$result_dir/$samplei/img_temp_monocular"

        python3 -W ignore dataset_vis_smplx_gemini.py \
            "$image_dir/$samplei/input_front" \
            "$result_dir/$samplei/hand_opt_front.npz" \
            "$temp_dir"

        python3 ~/image_to_video.py \
            "$temp_dir" \
            "$video_result_dir/${samplei}_monocular.mp4"

        rm -rf "$temp_dir"

    else
        echo "Skipping $samplei (contains $file_count files, expected exactly 10)."
    fi

done

echo "Done!"
