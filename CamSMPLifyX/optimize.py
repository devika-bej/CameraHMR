import os
import argparse
import numpy as np
import torch
from cam_smplifyx import SMPLifyX

def main(args):

    init_param_file = args.input
    image_base_dir = args.image_dir    
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    output_file_path =  os.path.join(args.output_dir, "output.npz")

    smplifyx = SMPLifyX(vis=args.vis, verbose=args.verbose)
    inp_data = np.load(init_param_file, allow_pickle=True)

    processed_data = {key: [] for key in inp_data}

    for i in range(len(inp_data['imgname'])):
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
        scale = torch.tensor(inp_data["scale"][i] / 200.0)
        dense_kp = inp_data["dense_kp"][i]

        # Run SMPLify optimization
        result = smplifyx(args, global_orient, body_pose, left_hand_pose, right_hand_pose, betas, cam_t, center, scale, cam_int, img_path, dense_kp, i)

        if result:
            processed_data["imgname"].append(img_path)
            processed_data["center"].append(inp_data["center"][i])
            processed_data["scale"].append(inp_data["scale"][i])
            processed_data["cam_int"].append(inp_data["cam_int"][i])
            # processed_data["gt_keypoints"].append(inp_data["gt_keypoints"][i])
            processed_data["gt_keypoints"] = []
            processed_data["cam_t"].append(result["camera_translation"].detach().cpu().numpy())
            processed_data["shape"].append(result["betas"][0].detach().cpu().numpy())

            body_pose = torch.hstack([result["global_orient"], result["pose"]]).detach().cpu().numpy()[0]
            processed_data["left_hand_pose"].append(result["lh_pose"][0].detach().cpu().numpy())
            processed_data["right_hand_pose"].append(result["rh_pose"][0].detach().cpu().numpy())
            processed_data["pose"].append(body_pose)
            
    # Save results
    np.savez(output_file_path, **processed_data)
    print(f"Processed data saved to {output_file_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run SMPLify on a dataset")
    parser.add_argument("--input", type=str, default='data/demo_files_for_optimization/init_params/filtered_aic.npz', help="Path to the initial parameter file (.npz)")
    parser.add_argument("--output_dir", type=str, default='out_params', help="Directory to save output data")
    parser.add_argument("--image_dir", type=str, default='data/demo_files_for_optimization/demo_images', help="Path to the image dataset directory")
    parser.add_argument("--vis", type=bool, required=False, help="Visualization of fitting")
    parser.add_argument("--verbose", type=bool, required=False, help="Print losses")
    parser.add_argument("--vis_int", type=int, default=100, required=False, help="Visualize result after every 100 iteration of optimization")
    parser.add_argument("--loss_cut", type=int, default=100, required=False, help="If initial loss is more than 100 we use high loss threshold else low loss threshold")
    parser.add_argument("--high_threshold", type=int, default=50, required=False, help="Loss threshold value to select the optimization result")
    parser.add_argument("--low_threshold", type=int, default=30, required=False, help="Loss threshold value to select the optimization result")

    args = parser.parse_args()
    main(args)

