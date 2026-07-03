#!/bin/bash

# Define output files
SAMPLE_RESULTS_FILE="sample_results_mono.csv"
BATCH_AVERAGES_FILE="batch_averages_mono.csv"

# Initialize output files with headers
echo "Batch,Sample,Opt_LHand_MSE,Opt_RHand_MSE,Base_LHand_MSE,Base_RHand_MSE" > "$SAMPLE_RESULTS_FILE"
echo "Batch,Total_Samples,Avg_Opt_LHand,Avg_Opt_RHand,Avg_Base_LHand,Avg_Base_RHand" > "$BATCH_AVERAGES_FILE"

# Variables to keep track of overall sums across all batches
overall_samples=0
overall_opt_lhand=0; overall_opt_rhand=0
overall_base_lhand=0; overall_base_rhand=0

echo "Starting MediaPipe evaluation loop..."

# Loop through batches 1 to 9
for batch in {1..9}; do
    batch_dir="../NGT_Temporary_${batch}"
    
    if [ ! -d "$batch_dir" ]; then
        echo "Directory $batch_dir not found. Skipping..."
        continue
    fi

    # Variables for batch-level sums
    batch_samples=0
    batch_opt_lhand=0; batch_opt_rhand=0
    batch_base_lhand=0; batch_base_rhand=0

    echo "Processing Batch ${batch}..."

    # Loop through all sample directories in the current batch
    for sample_dir in "$batch_dir"/sample*; do
        if [ ! -d "$sample_dir" ]; then
            continue
        fi

        sample_name=$(basename "$sample_dir")
        
        # Define the exact paths based on your new directory structure
        opt_file="${sample_dir}/hand_front/output.npz"
        base_file="${sample_dir}/smplx_front/mesh_estimation_output.npz"

        # Check if required files exist
        if [[ -f "$opt_file" && -f "$base_file" ]]; then
            
            # --- 1. RUN OPTIMIZATION EVALUATION ---
            # Compare optimized estimate vs target keypoints contained in base_file
            opt_out=$(python3 -W ignore mediapipe_metric.py "$opt_file" "$base_file" 2>&1)
            o_lhand=$(echo "$opt_out" | grep "Mean L-Hand MSE:" | awk '{print $4}')
            o_rhand=$(echo "$opt_out" | grep "Mean R-Hand MSE:" | awk '{print $4}')

            # --- 2. RUN BASELINE EVALUATION ---
            # Compare base estimate vs target keypoints contained in base_file
            base_out=$(python3 -W ignore mediapipe_metric.py "$base_file" "$base_file" 2>&1)
            b_lhand=$(echo "$base_out" | grep "Mean L-Hand MSE:" | awk '{print $4}')
            b_rhand=$(echo "$base_out" | grep "Mean R-Hand MSE:" | awk '{print $4}')

            # SAFETY CHECK: Ensure the python script didn't crash on either run
            if [[ -z "$o_lhand" || -z "$b_lhand" ]]; then
                echo "  [Error] Python script failed or metrics missing for ${sample_name}. Skipping."
                continue
            fi

            # Write the sample-wise results
            echo "${batch},${sample_name},${o_lhand},${o_rhand},${b_lhand},${b_rhand}" >> "$SAMPLE_RESULTS_FILE"

            # Add to batch sums (using awk to handle float math)
            batch_opt_lhand=$(awk "BEGIN {print $batch_opt_lhand + $o_lhand; exit}")
            batch_opt_rhand=$(awk "BEGIN {print $batch_opt_rhand + $o_rhand; exit}")
            
            batch_base_lhand=$(awk "BEGIN {print $batch_base_lhand + $b_lhand; exit}")
            batch_base_rhand=$(awk "BEGIN {print $batch_base_rhand + $b_rhand; exit}")
            
            batch_samples=$((batch_samples + 1))

            # Add to overall sums
            overall_opt_lhand=$(awk "BEGIN {print $overall_opt_lhand + $o_lhand; exit}")
            overall_opt_rhand=$(awk "BEGIN {print $overall_opt_rhand + $o_rhand; exit}")
            
            overall_base_lhand=$(awk "BEGIN {print $overall_base_lhand + $b_lhand; exit}")
            overall_base_rhand=$(awk "BEGIN {print $overall_base_rhand + $b_rhand; exit}")
            
            overall_samples=$((overall_samples + 1))
        else
            echo "  [Warning] Missing Opt or Base file in ${sample_dir}, skipping."
        fi
    done

    # Calculate and write batch-wise averages
    if [ "$batch_samples" -gt 0 ]; then
        # Formatting floats to 6 decimal places to match python script
        avg_o_lhand=$(awk "BEGIN {printf \"%.6f\", $batch_opt_lhand / $batch_samples; exit}")
        avg_o_rhand=$(awk "BEGIN {printf \"%.6f\", $batch_opt_rhand / $batch_samples; exit}")
        
        avg_b_lhand=$(awk "BEGIN {printf \"%.6f\", $batch_base_lhand / $batch_samples; exit}")
        avg_b_rhand=$(awk "BEGIN {printf \"%.6f\", $batch_base_rhand / $batch_samples; exit}")

        echo "${batch},${batch_samples},${avg_o_lhand},${avg_o_rhand},${avg_b_lhand},${avg_b_rhand}" >> "$BATCH_AVERAGES_FILE"
        echo "  -> Evaluated $batch_samples samples for Batch $batch."
    else
        echo "  -> No valid samples found for Batch $batch."
    fi
done

# Calculate and print the overall average across all batches and samples
echo "=========================================================="
if [ "$overall_samples" -gt 0 ]; then
    oa_o_lhand=$(awk "BEGIN {printf \"%.6f\", $overall_opt_lhand / $overall_samples; exit}")
    oa_o_rhand=$(awk "BEGIN {printf \"%.6f\", $overall_opt_rhand / $overall_samples; exit}")
    
    oa_b_lhand=$(awk "BEGIN {printf \"%.6f\", $overall_base_lhand / $overall_samples; exit}")
    oa_b_rhand=$(awk "BEGIN {printf \"%.6f\", $overall_base_rhand / $overall_samples; exit}")

    echo "ALL BATCHES COMPLETE - OVERALL AVERAGES ($overall_samples Samples)"
    echo "----------------------------------------------------------"
    echo "METRIC         | BASELINE AVG | OPTIMIZATION AVG "
    echo "----------------------------------------------------------"
    echo "Left Hand MSE  | $oa_b_lhand       | $oa_o_lhand"
    echo "Right Hand MSE | $oa_b_rhand       | $oa_o_rhand"
else
    echo "No samples were successfully evaluated across any batches."
fi
echo "=========================================================="
echo "Results saved to $SAMPLE_RESULTS_FILE and $BATCH_AVERAGES_FILE"
