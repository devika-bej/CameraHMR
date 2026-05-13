import os
import argparse
import numpy as np
import torch
from hand_smplifyx import HandOptimizer
from hand_vertex_testing import check_coordinate_alignment

CUDA_LAUNCH_BLOCKING=1

def main(args):
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    output_file_path = os.path.join(args.output_dir, "output.npz")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    hand_refiner = HandOptimizer()
    
    global_orients = []
    body_poses = []
    left_hand_poses = []
    right_hand_poses = []
    betas = []
    cam_ints = []
    cam_ts = []
    
    is_front = True
    final_data = {}
    
    for init_param_file in args.inputs:
        inp_data = np.load(init_param_file, allow_pickle=True)
        inp_data = dict(inp_data)
        if is_front:
            for key in inp_data.keys():
                if key not in ["global_orient", "body_pose", "left_hand_pose", "right_hand_pose", "betas"]:
                    final_data[key] = inp_data[key]
            is_front = False
        global_orients.append(inp_data["global_orient"])
        body_poses.append(inp_data["body_pose"])
        left_hand_poses.append(inp_data["left_hand_pose"])
        right_hand_poses.append(inp_data["right_hand_pose"])
        betas.append(inp_data["shape"])
        cam_ints.append(inp_data["cam_int"])
        cam_ts.append(inp_data["cam_t"])
    global_orients = torch.tensor(global_orients)
    body_poses = torch.tensor(body_poses)
    left_hand_poses = torch.tensor(left_hand_poses)
    right_hand_poses = torch.tensor(right_hand_poses)
    betas = torch.tensor(betas)
    cam_ints = torch.tensor(cam_ints)
    cam_ts = torch.tensor(cam_ts)
    
    print("Multiview Stitching")
    global_orient, body_pose, left_hand, right_hand, betas = hand_refiner.stitch(
        global_orients, body_poses, left_hand_poses, right_hand_poses, betas, 
        cam_ints, cam_ts)
    final_data["global_orient"] = global_orient
    final_data["body_pose"] = body_pose
    final_data["left_hand_pose"] = left_hand
    final_data["right_hand_pose"] = right_hand
    final_data["shape"] = betas

    output_file_path = os.path.join(args.output_dir, "output_with_stitching.npz")
    # Save results# Convert any remaining tensors in the lists to numpy arrays
    for key in final_data:
        final_data[key] = [
            item.detach().cpu().numpy() if isinstance(item, torch.Tensor) else item 
            for item in final_data[key]
        ]
    np.savez(output_file_path, **final_data)
    print(f"Processed data saved to {output_file_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Hand Optimization on a dataset")
    parser.add_argument(
        "--inputs",
        nargs="+",
        default="data/demo_files_for_optimization/init_params/filtered_aic.npz",
        help="Path to the initial parameter file (.npz)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="out_params",
        help="Directory to save output data",
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