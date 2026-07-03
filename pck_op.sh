#!/bin/bash

# Define the master output file for OpenPose results
MASTER_CSV="global_joint_distances_openpose.csv"

# Clear the file if it already exists so we don't append to old data
if [ -f "$MASTER_CSV" ]; then
    rm "$MASTER_CSV"
fi

echo "Starting Global Joint Distance Extraction for PCK (OpenPose)..."

overall_samples=0

# Loop through batches 1 to 9
for batch in {1..9}; do
    batch_dir="../NGT_Temporary_${batch}"
    
    if [ ! -d "$batch_dir" ]; then
        echo "Directory $batch_dir not found. Skipping..."
        continue
    fi

    echo "Processing Batch ${batch}..."
    batch_samples=0

    # Loop through all sample directories in the current batch
    for sample_dir in "$batch_dir"/sample*; do
        if [ ! -d "$sample_dir" ]; then
            continue
        fi

        sample_name=$(basename "$sample_dir")
        
        # --- IMPORTANT: VERIFY THESE PATHS MATCH YOUR DIRECTORY STRUCTURE ---
        MULTI_FILE="${sample_dir}/multiview/output_with_stitching.npz"         
        MONO_FILE="${sample_dir}/hand_front/output.npz"
        BASE_FILE="${sample_dir}/smplx_front/mesh_estimation_output.npz" 
        
        # Update this variable to point directly to the file holding your OpenPose NPZ arrays
        OP_FILE="${sample_dir}/openpose.npz" 
        # ------------------------------------------------------------------

        # Check if ALL required files exist before running Python
        if [[ -f "$MULTI_FILE" && -f "$MONO_FILE" && -f "$BASE_FILE" && -f "$OP_FILE" ]]; then
            
            # Run the Python script to extract distances and append to MASTER_CSV
            python3 -W ignore eval_pck_op.py \
                "$batch" \
                "$sample_name" \
                "$MULTI_FILE" \
                "$MONO_FILE" \
                "$BASE_FILE" \
                "$OP_FILE" \
                "$MASTER_CSV"

            # Check if the python script executed successfully
            if [ $? -eq 0 ]; then
                batch_samples=$((batch_samples + 1))
                overall_samples=$((overall_samples + 1))
            else
                echo "  [Error] Python script failed for ${sample_name}. Skipping."
            fi
            
        else
            echo "  [Warning] Missing one or more .npz files in ${sample_dir}, skipping."
        fi
    done
    
    echo "  -> Extracted data from $batch_samples samples in Batch $batch."
done

echo "=========================================================="
echo "Extraction Complete!"
echo "Total samples successfully processed: $overall_samples"
echo "Master dataset saved to: $MASTER_CSV"
echo "=========================================================="
