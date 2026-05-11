import os
import argparse
import numpy as np
import torch
from test import HandOptimizer
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
    # inp_data = dict(inp_data)

    processed_data = {key: [] for key in inp_data}

    for i in range(len(inp_data["imgname"])):
        img_path = os.path.join(image_base_dir, inp_data["imgname"][i])
        print(f"Processing: {img_path}")

        if not os.path.exists(img_path):
            print(f"File not found: {img_path}")
            continue

        # Extract initial data and convert necessary inputs to PyTorch tensors
        global_orient = torch.tensor(np.expand_dims(inp_data["global_orient"][i], axis=0)).to(device).float()
        cam_int_np = inp_data["cam_int"][i]
        cam_t_np = inp_data["cam_t"][i]
        center = torch.tensor(inp_data["center"][i]).to(device).float()
        scale = torch.tensor(inp_data["scale"][i]).to(device).float()

        # Tensors required for hand refinement
        body_pose = torch.tensor(np.expand_dims(inp_data["body_pose"][i], axis=0)).to(device).float()
        left_hand_pose = torch.tensor(np.expand_dims(inp_data["left_hand_pose"][i], axis=0)).to(device).float()
        right_hand_pose = torch.tensor(np.expand_dims(inp_data["right_hand_pose"][i], axis=0)).to(device).float()
        betas = torch.tensor(np.expand_dims(inp_data["shape"][i], axis=0)).to(device).float()
        
        c_int = torch.tensor(cam_int_np).unsqueeze(0).to(device).float()
        c_t = torch.tensor(cam_t_np).to(device).float()
        
        mediapipe_kp_left = inp_data["mediapipe_kp_left"][i]
        mediapipe_kp_right = inp_data["mediapipe_kp_right"][i]

        # Populate unoptimized body variables directly from initial data
        processed_data["imgname"].append(img_path)
        processed_data["center"].append(inp_data["center"][i])
        processed_data["scale"].append(inp_data["scale"][i])
        processed_data["cam_int"].append(cam_int_np)
        processed_data["cam_t"].append(cam_t_np)
        processed_data["shape"].append(inp_data["shape"][i])
        processed_data["global_orient"].append(global_orient[0])
        processed_data["body_pose"].append(body_pose[0])

        left_hand_pose, right_hand_pose, left_wrist, right_wrist = hand_refiner.refine(
            global_orient=global_orient,
            body_pose=body_pose,
            left_hand_pose=left_hand_pose,
            right_hand_pose=right_hand_pose,
            betas=betas,
            cam_t=c_t,
            cam_int=c_int,
            bbox_center=center,
            bbox_scale=scale,
            target_mp=torch.tensor(np.stack([mediapipe_kp_right, mediapipe_kp_left])).to(device).float(),
            img_path=img_path
        )
        
        processed_data["left_hand_pose"].append(left_hand_pose[0].detach().cpu().numpy())
        processed_data["right_hand_pose"].append(right_hand_pose[0].detach().cpu().numpy())
        processed_data["body_pose"][-1][19] = left_wrist
        processed_data["body_pose"][-1][20] = right_wrist
    
    # Save results# Convert any remaining tensors in the lists to numpy arrays
    for key in processed_data:
        processed_data[key] = [
            item.detach().cpu().numpy() if isinstance(item, torch.Tensor) else item 
            for item in processed_data[key]
        ]
    np.savez(output_file_path, **processed_data)
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
