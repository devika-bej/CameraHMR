#!/bin/bash

# Define output files
SAMPLE_RESULTS_FILE="temporal_sample_results.csv"
BATCH_AVERAGES_FILE="temporal_batch_averages.csv"

# Initialize output files with headers
echo "Batch,Sample,CameraHMR_Mean_Jitter,Base_Mean_Jitter,Stitched_Mean_Jitter" > "$SAMPLE_RESULTS_FILE"
echo "Batch,Total_Samples,Avg_CameraHMR_Jitter,Avg_Base_Jitter,Avg_Stitched_Jitter" > "$BATCH_AVERAGES_FILE"

# Variables to keep track of overall sums across all batches
overall_samples=0
overall_cam_jitter=0
overall_base_jitter=0
overall_stitch_jitter=0

echo "Starting 3-way temporal evaluation loop..."

# Loop through batches 1 to 9
for batch in {1..9}; do
    batch_dir="../NGT_Temporary_${batch}"
    
    if [ ! -d "$batch_dir" ]; then
        echo "Directory $batch_dir not found. Skipping..."
        continue
    fi

    # Variables for batch-level sums
    batch_samples=0
    batch_cam_jitter=0
    batch_base_jitter=0
    batch_stitch_jitter=0

    echo "Processing Batch ${batch}..."

    # Loop through all sample directories in the current batch
    for sample_dir in "$batch_dir"/sample*; do
        if [ ! -d "$sample_dir" ]; then
            continue
        fi

        sample_name=$(basename "$sample_dir")
        
        # Define paths for the temporal evaluation (UPDATE THE CAMERHMR PATH AS NEEDED)
        base_file="${sample_dir}/hand_front/output.npz" 
        cam_file="${sample_dir}/smplx_front/mesh_estimation_output.npz"
        stitch_file="${sample_dir}/multiview/output_with_stitching.npz"

        # Check if ALL THREE required files exist
        if [[ -f "$cam_file" && -f "$base_file" && -f "$stitch_file" ]]; then
            
            # Run the python evaluation script and capture output 
            output=$(python3 -W ignore temporal_metric.py "$cam_file" "$base_file" "$stitch_file" 2>&1)

            # Extract the Mean Jitter values using grep and awk (extracts the 6th word in the string)
            c_jitter=$(echo "$output" | grep "CameraHMR Mean Jitter" | awk '{print $6}')
            b_jitter=$(echo "$output" | grep "Baseline Mean Jitter" | awk '{print $6}')
            s_jitter=$(echo "$output" | grep "Stitched Mean Jitter" | awk '{print $6}')

            # SAFETY CHECK: Ensure the python script didn't crash and all 3 metrics were found
            if [[ -z "$c_jitter" || -z "$b_jitter" || -z "$s_jitter" ]]; then
                echo "  [Error] Python script failed or metrics missing for ${sample_name}. Skipping."
                continue
            fi

            # Write the sample-wise results
            echo "${batch},${sample_name},${c_jitter},${b_jitter},${s_jitter}" >> "$SAMPLE_RESULTS_FILE"

            # Add to batch sums (using awk to handle float math)
            batch_cam_jitter=$(awk "BEGIN {print $batch_cam_jitter + $c_jitter; exit}")
            batch_base_jitter=$(awk "BEGIN {print $batch_base_jitter + $b_jitter; exit}")
            batch_stitch_jitter=$(awk "BEGIN {print $batch_stitch_jitter + $s_jitter; exit}")
            
            batch_samples=$((batch_samples + 1))

            # Add to overall sums
            overall_cam_jitter=$(awk "BEGIN {print $overall_cam_jitter + $c_jitter; exit}")
            overall_base_jitter=$(awk "BEGIN {print $overall_base_jitter + $b_jitter; exit}")
            overall_stitch_jitter=$(awk "BEGIN {print $overall_stitch_jitter + $s_jitter; exit}")
            
            overall_samples=$((overall_samples + 1))
        else
            echo "  [Warning] Missing CameraHMR, Baseline, or Stitched files in ${sample_dir}, skipping."
        fi
    done

    # Calculate and write batch-wise averages
    if [ "$batch_samples" -gt 0 ]; then
        avg_c_jitter=$(awk "BEGIN {printf \"%.6f\", $batch_cam_jitter / $batch_samples; exit}")
        avg_b_jitter=$(awk "BEGIN {printf \"%.6f\", $batch_base_jitter / $batch_samples; exit}")
        avg_s_jitter=$(awk "BEGIN {printf \"%.6f\", $batch_stitch_jitter / $batch_samples; exit}")

        echo "${batch},${batch_samples},${avg_c_jitter},${avg_b_jitter},${avg_s_jitter}" >> "$BATCH_AVERAGES_FILE"
        echo "  -> Evaluated $batch_samples samples for Batch $batch."
    else
        echo "  -> No valid samples found for Batch $batch."
    fi
done

# Calculate and print the overall average across all batches and samples
echo "====================================================================="
if [ "$overall_samples" -gt 0 ]; then
    oa_c_jitter=$(awk "BEGIN {printf \"%.6f\", $overall_cam_jitter / $overall_samples; exit}")
    oa_b_jitter=$(awk "BEGIN {printf \"%.6f\", $overall_base_jitter / $overall_samples; exit}")
    oa_s_jitter=$(awk "BEGIN {printf \"%.6f\", $overall_stitch_jitter / $overall_samples; exit}")

    echo "ALL TEMPORAL BATCHES COMPLETE ($overall_samples Samples)"
    echo "---------------------------------------------------------------------"
    echo "METRIC                   | CAMERHMR AVG | BASELINE AVG | STITCHED AVG"
    echo "---------------------------------------------------------------------"
    echo "Mean Jitter (Abs Accel)  | $oa_c_jitter     | $oa_b_jitter     | $oa_s_jitter"
else
    echo "No samples were successfully evaluated across any batches."
fi
echo "====================================================================="
echo "Results saved to $SAMPLE_RESULTS_FILE and $BATCH_AVERAGES_FILE"
