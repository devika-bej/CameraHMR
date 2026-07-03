#!/bin/bash

# Check if the user provided the input number x
if [ -z "$1" ]; then
    echo "Usage: $0 <x>"
    exit 1
fi

x=$1
base_dir="../NGT_Temporary_${x}/"

# Verify that the base directory actually exists
if [ ! -d "$base_dir" ]; then
    echo "Error: Base directory '$base_dir' does not exist."
    exit 1
fi

# Loop through each sample directory inside base_dir
# The trailing slash ensures it only matches directories
for sample_dir in "${base_dir}"sample*/; do
    
    # Double-check that it is a directory (handles cases where no match is found)
    if [ -d "$sample_dir" ]; then
        
        # Define input and output paths
        input_folder="${sample_dir}input_front/"
        out_folder="${sample_dir}cropped_front/"
        
        echo "Processing: $sample_dir"
        
        # Execute the python script
        python3 -W ignore demo_detectron.py \
            --img_folder "$input_folder" \
            --out_folder "$out_folder"
            
    fi
done

echo "Processing complete."
