import os
import argparse
import numpy as np
import torch
from cam_smplifyx import SMPLifyX
from constants import ALL_MODEL_DIR
from hand_smplifyx import HandOptimizer
from hand_vertex_testing import check_coordinate_alignment

CUDA_LAUNCH_BLOCKING=1


def main(args):

    init_param_file = args.input
    image_base_dir = args.image_dir
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    output_file_path = os.path.join(args.output_dir, "output.npz")

    smplifyx = SMPLifyX(vis=args.vis, verbose=args.verbose)
    hand_refiner = HandOptimizer(model_path=ALL_MODEL_DIR)
    inp_data = np.load(init_param_file, allow_pickle=True)

    processed_data = {key: [] for key in inp_data}

    for i in range(len(inp_data["imgname"])):
        img_path = os.path.join(image_base_dir, inp_data["imgname"][i])
        print(f"Processing: {img_path}")

        if not os.path.exists(img_path):
            print(f"File not found: {img_path}")
            continue

        # Extract data
        global_orient = inp_data["global_orient"][i]
        body_pose = np.expand_dims(inp_data["body_pose"][i], axis=0)
        left_hand_pose = np.expand_dims(inp_data["left_hand_pose"][i], axis=0)
        right_hand_pose = np.expand_dims(inp_data["right_hand_pose"][i], axis=0)
        betas = np.expand_dims(inp_data["shape"][i], axis=0)
        cam_int = torch.tensor(inp_data["cam_int"][i])
        cam_t = torch.tensor(inp_data["cam_t"][i])
        center = torch.tensor(inp_data["center"][i])
        scale = torch.tensor(inp_data["scale"][i])
        dense_kp = inp_data["dense_kp"][i]
        mediapipe_kp_left = inp_data["mediapipe_kp_left"][i]
        mediapipe_kp_right = inp_data["mediapipe_kp_right"][i]

        # Run SMPLify optimization
        result = smplifyx(
            args,
            global_orient,
            body_pose,
            left_hand_pose,
            right_hand_pose,
            betas,
            cam_t,
            center,
            scale,
            cam_int,
            img_path,
            dense_kp=dense_kp,
            ind=i,
        )

        if result:
            processed_data["imgname"].append(img_path)
            processed_data["center"].append(inp_data["center"][i])
            processed_data["scale"].append(inp_data["scale"][i])
            processed_data["cam_int"].append(inp_data["cam_int"][i])
            # processed_data["gt_keypoints"].append(inp_data["gt_keypoints"][i])
            processed_data["gt_keypoints"] = []
            processed_data["cam_t"].append(
                result["camera_translation"].detach().cpu().numpy()
            )
            processed_data["shape"].append(result["betas"][0].detach().cpu().numpy())
            processed_data["left_hand_pose"].append(
                result["lh_pose"][0].detach().cpu().numpy()
            )
            processed_data["right_hand_pose"].append(
                result["rh_pose"][0].detach().cpu().numpy()
            )
            processed_data["body_pose"].append(
                result["pose"][0].detach().cpu().numpy()
            )
            processed_data["global_orient"].append(
                result["global_orient"].detach().cpu().numpy()
            )
        
        if result:
            # --- START HAND REFINEMENT ---
            device = result["lh_pose"].device
            # Format camera params for the refiner
            c_int = cam_int.unsqueeze(0).to(device).float()
            c_t = result["camera_translation"].to(device).float()

            # Process Left Hand
            if len(mediapipe_kp_left) > 0:
                print("optimizing left hand...")
                # MediaPipe gives multiple hands; we take the first detection
                mp_left = torch.tensor(np.expand_dims(mediapipe_kp_left, axis=0)).to(device).float()
                
                # Combine Wrist (Joint 20) and LH Pose (15 joints)
                l_init = torch.cat([result["pose"][:, 19:20, :], result["lh_pose"]], dim=1)
                
                # check = check_coordinate_alignment(
                #     hand_refiner, mp_left, l_init, result["betas"][:, :10], c_int, c_t, center, scale, is_left=True,
                #     image_path=img_path, # Assuming scale is related to image size
                #     save_overlay=True, overlay_path=f"coord_check_{i}_left.png"
                # )
                # print("Coordinate alignment check completed. Overlay saved as:", f"coord_check_{i}_left.png")
                refined_l_pose, _ = hand_refiner.refine(
                    mp_left, l_init, result["betas"][:, :10], c_int, c_t, center, scale, is_left=True, inp_image_path=img_path
                )
                
                result["pose"] = result["pose"].clone() # Clone to avoid in-place modification
                result["lh_pose"] = result["lh_pose"].clone() # Clone to avoid in-place modification
                
                # Update the main result dictionary
                refined_l_pose = refined_l_pose.reshape(1, 16, 3) # Reshape back to (1, 16, 3)
                result["pose"][:, 19:20, :] = refined_l_pose[:, :1, :] # Update Wrist
                result["lh_pose"] = refined_l_pose[:, 1:, :]           # Update Fingers

            # Process Right Hand
            if len(mediapipe_kp_right) > 0:
                print("optimizing right hand...")
                mp_right = torch.tensor(np.expand_dims(mediapipe_kp_right, axis=0)).to(device).float()

                # Combine Wrist (Joint 21) and RH Pose (15 joints)
                r_init = torch.cat([result["pose"][:, 20:21, :], result["rh_pose"]], dim=1)
                
                # check = check_coordinate_alignment(
                #     hand_refiner, mp_right, r_init, result["betas"][:, :10], c_int, c_t, center, scale, is_left=False,
                #     image_path=img_path, # Assuming scale is related to image size
                #     save_overlay=True, overlay_path=f"coord_check_{i}_right.png"
                # )
                # print("Coordinate alignment check completed. Overlay saved as:", f"coord_check_{i}_right.png")
                refined_r_pose, _ = hand_refiner.refine(
                    mp_right, r_init, result["betas"][:, :10], c_int, c_t, center, scale, is_left=False, inp_image_path=img_path
                )
                
                result["pose"] = result["pose"].clone() # Clone to avoid in-place modification
                result["rh_pose"] = result["rh_pose"].clone() # Clone to avoid in-place modification
                
                refined_r_pose = refined_r_pose.reshape(1, 16, 3) # Reshape back to (1, 16, 3)
                result["pose"][:, 20:21, :] = refined_r_pose[:, :1, :] # Update Wrist
                result["rh_pose"] = refined_r_pose[:, 1:, :]           # Update Fingers
            # --- END HAND REFINEMENT ---
            
            processed_data["body_pose"][-1] = result["pose"][0].detach().cpu().numpy()
            processed_data["left_hand_pose"][-1] = result["lh_pose"][0].detach().cpu().numpy()
            processed_data["right_hand_pose"][-1] = result["rh_pose"][0].detach().cpu().numpy()

            
    # Save results
    np.savez(output_file_path, **processed_data)
    print(f"Processed data saved to {output_file_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run SMPLify on a dataset")
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
    parser.add_argument(
        "--vis", type=bool, required=False, help="Visualization of fitting"
    )
    parser.add_argument("--verbose", type=bool, required=False, help="Print losses")
    parser.add_argument(
        "--vis_int",
        type=int,
        default=100,
        required=False,
        help="Visualize result after every 100 iteration of optimization",
    )
    parser.add_argument(
        "--loss_cut",
        type=int,
        default=100,
        required=False,
        help="If initial loss is more than 100 we use high loss threshold else low loss threshold",
    )
    parser.add_argument(
        "--high_threshold",
        type=int,
        default=50,
        required=False,
        help="Loss threshold value to select the optimization result",
    )
    parser.add_argument(
        "--low_threshold",
        type=int,
        default=30,
        required=False,
        help="Loss threshold value to select the optimization result",
    )

    args = parser.parse_args()
    main(args)