#!/bin/bash

# Define the single master output file
MASTER_CSV="global_joint_distances.csv"

# Clear the file if it already exists so we don't append to old data
if [ -f "$MASTER_CSV" ]; then
    rm "$MASTER_CSV"
fi

echo "Starting Global PCK Data Extraction..."

# Loop through batches 1 to 9
for batch in {1..9}; do
    batch_dir="../NGT_Temporary_${batch}"
    
    if [ ! -d "$batch_dir" ]; then
        echo "Directory $batch_dir not found. Skipping..."
        continue
    fi

    echo "Processing Batch ${batch}..."

    # Loop through all sample directories in the current batch
    for sample_dir in "$batch_dir"/sample*; do
        if [ ! -d "$sample_dir" ]; then
            continue
        fi

        sample_name=$(basename "$sample_dir")
        
        # --- IMPORTANT: VERIFY THESE PATHS MATCH YOUR DIRECTORY STRUCTURE ---
        MULTI_FILE="${sample_dir}/multiview/output_with_stitching.npz"         # Your Multi-view output
        MONO_FILE="${sample_dir}/hand_front/output.npz"    # Your Monocular output
        BASE_FILE="${sample_dir}/smplx_front/mesh_estimation_output.npz" # CameraHMR Base
        MP_FILE="${sample_dir}/smplx_front/mesh_estimation_output.npz"   # File containing MediaPipe GT
        # ------------------------------------------------------------------

        # Check if ALL required files exist before running Python
        if [[ -f "$MULTI_FILE" && -f "$MONO_FILE" && -f "$BASE_FILE" && -f "$MP_FILE" ]]; then
            
            # Pass everything to the Python script to process and append to the CSV
            python3 -W ignore eval_pck.py \
                "$batch" \
                "$sample_name" \
                "$MULTI_FILE" \
                "$MONO_FILE" \
                "$BASE_FILE" \
                "$MP_FILE" \
                "$MASTER_CSV"
                
        else
            echo "  [Warning] Missing one or more .npz files in ${sample_dir}, skipping."
        fi
    done
done

echo "=========================================================="
echo "Extraction Complete! Master dataset saved to $MASTER_CSV."
echo "You can now use this CSV to plot the standard global PCK curve."
echo "=========================================================="
