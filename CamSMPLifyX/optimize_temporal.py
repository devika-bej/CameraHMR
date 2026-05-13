import os
import argparse
import numpy as np
import torch
from hand_smplifyx import HandOptimizer
from hand_vertex_testing import check_coordinate_alignment

CUDA_LAUNCH_BLOCKING=1

def main(args):
    init_param_file = args.input
    image_base_dir = args.image_dir
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    output_file_path = os.path.join(args.output_dir, "output.npz")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize only the Hand Optimizer
    hand_refiner = HandOptimizer()
    inp_data = np.load(init_param_file, allow_pickle=True)
    inp_data = dict(inp_data)

    print("Temporal smoothing")
    left_hand, right_hand = hand_refiner.smoothen(torch.tensor(inp_data["left_hand_pose"]), torch.tensor(inp_data["right_hand_pose"]))
    inp_data["left_hand_pose"] = left_hand
    inp_data["right_hand_pose"] = right_hand

    output_file_path = os.path.join(args.output_dir, "output_with_temporal.npz")
    # Save results# Convert any remaining tensors in the lists to numpy arrays
    for key in inp_data:
        inp_data[key] = [
            item.detach().cpu().numpy() if isinstance(item, torch.Tensor) else item 
            for item in inp_data[key]
        ]
    np.savez(output_file_path, **inp_data)
    print(f"Processed data saved to {output_file_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Hand Optimization on a dataset")
    parser.add_argument(
        "--input",
        type=str,
        default="data/demo_files_for_optimization/init_params/filtered_aic.npz",
        help="Path to the initial parameter file (.npz)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="out_params",
        help="Directory to save output data",
    )
    parser.add_argument(
        "--image_dir",
        type=str,
        default="data/demo_files_for_optimization/demo_images",
        help="Path to the image dataset directory",
    )
    # Note: Kept the arguments from your original parser to prevent breaking your terminal commands,
    # though vis/vis_int/loss bounds aren't used for the hand refiner in the provided snippet.
    parser.add_argument("--vis", type=bool, required=False, help="Visualization of fitting")
    parser.add_argument("--verbose", type=bool, required=False, help="Print losses")
    parser.add_argument("--vis_int", type=int, default=100, required=False)
    parser.add_argument("--loss_cut", type=int, default=100, required=False)
    parser.add_argument("--high_threshold", type=int, default=50, required=False)
    parser.add_argument("--low_threshold", type=int, default=30, required=False)

    args = parser.parse_args()
    main(args)