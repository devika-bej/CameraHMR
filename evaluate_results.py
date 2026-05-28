import numpy as np
import sys
import torch
import cv2

# Import the SMPLX model wrapper with OpenPose mapping
from CamSMPLifyX.utils.smplx_openpose import SMPLX_
# Import projection and HandOptimizer utilities
from CamSMPLifyX.hand_smplifyx import perspective_projection, j2d_processing, HandOptimizer
from CamSMPLifyX.constants import SMPLX_MODEL_DIR, NUM_BETAS_SMPLX

def evaluate(estimate, openpose):
    n_frames = estimate['imgname'].shape[0]
    body_errs = []
    lhand_errs = []
    rhand_errs = []
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Initialize the SMPLX_ model with OpenPose joint regressor
    model = SMPLX_(
        SMPLX_MODEL_DIR,
        num_betas=NUM_BETAS_SMPLX,
        use_pca=False
    ).to(device)
    
    # We will use the HandOptimizer to extract proper MANO/OpenPose hand joints from vertices and joints
    hand_opt = HandOptimizer(device=device)
    
    for i in range(n_frames):
        # Shape: (N_joints, 3) where the 3rd column is confidence
        op_body = openpose['body_pose'][i]
        op_lhand = openpose['left_hand_pose'][i]
        op_rhand = openpose['right_hand_pose'][i]
        
        # 1. Get SMPLX parameters for the current frame
        global_orient = torch.tensor(estimate['global_orient'][i:i+1]).to(device).float()
        body_pose = torch.tensor(estimate['body_pose'][i:i+1]).to(device).float()
        left_hand_pose = torch.tensor(estimate['left_hand_pose'][i:i+1]).to(device).float()
        right_hand_pose = torch.tensor(estimate['right_hand_pose'][i:i+1]).to(device).float()
        betas = torch.tensor(estimate['shape'][i:i+1]).to(device).float()
        
        cam_t = torch.tensor(estimate['cam_t'][i]).to(device).float()
        cam_int = torch.tensor(estimate['cam_int'][i]).to(device).float()
        
        # 2. Forward pass through SMPLX
        with torch.no_grad():
            smplx_out = model(
                global_orient=global_orient,
                body_pose=body_pose,
                left_hand_pose=left_hand_pose,
                right_hand_pose=right_hand_pose,
                betas=betas
            )
            
            # --- Get Body Joints ---
            # smplx_openpose maps output.body_joints to the 25 OpenPose joints automatically
            body_3d = smplx_out.joints[0, :25] 
            
            # --- Get Hand Joints using hand_smplifyx logic ---
            left_wrist_joint = smplx_out.joints[:, 20]
            right_wrist_joint = smplx_out.joints[:, 21]
            lh_joints = smplx_out.joints[:, 25:40, :]
            rh_joints = smplx_out.joints[:, 40:55, :]
            
            estimate_3d_lh, estimate_3d_rh = hand_opt.get_mano_landmarks(
                left_wrist_joint,
                right_wrist_joint,
                lh_joints,
                rh_joints,
                smplx_out.vertices
            )
            
            # 3. Project 3D points to 2D
            est_body_2d = perspective_projection(body_3d, cam_t, cam_int)
            est_lhand_2d = perspective_projection(estimate_3d_lh[0], cam_t, cam_int)
            est_rhand_2d = perspective_projection(estimate_3d_rh[0], cam_t, cam_int)
            
            # If the estimate uses cropping (bbox_center / bbox_scale exist), project to crop space.
            if 'bbox_center' in estimate and 'bbox_scale' in estimate:
                bbox_center = torch.tensor(estimate['bbox_center'][i]).to(device).float()
                bbox_scale = torch.tensor(estimate['bbox_scale'][i]).to(device).float()
                
                est_body_2d = j2d_processing(est_body_2d[:, :2], bbox_center, bbox_scale)
                est_lhand_2d = j2d_processing(est_lhand_2d[:, :2], bbox_center, bbox_scale)
                est_rhand_2d = j2d_processing(est_rhand_2d[:, :2], bbox_center, bbox_scale)
            else:
                est_body_2d = est_body_2d[:, :2]
                est_lhand_2d = est_lhand_2d[:, :2]
                est_rhand_2d = est_rhand_2d[:, :2]
            
            # Convert tensors back to numpy for evaluation
            est_body_2d = est_body_2d.cpu().numpy()
            est_lhand_2d = est_lhand_2d.cpu().numpy()
            est_rhand_2d = est_rhand_2d.cpu().numpy()
            
            # 4. Calculate MSE separately for body, left hand, right hand (using valid keypoints)
            def compute_mse(pred, target):
                # valid_mask = target[:, 2] > 0.0  # OpenPose confidence > 0
                # if np.sum(valid_mask) == 0:
                #     return None
                return np.mean((pred - target) ** 2)
            
            err_body = compute_mse(est_body_2d, op_body)
            if err_body is not None:
                body_errs.append(err_body)
                
            err_lhand = compute_mse(est_lhand_2d, op_lhand)
            if err_lhand is not None:
                lhand_errs.append(err_lhand)
                
            err_rhand = compute_mse(est_rhand_2d, op_rhand)
            if err_rhand is not None:
                rhand_errs.append(err_rhand)

    # Print overall statistics
    print(f"Evaluation over {n_frames} frames:")
    print(f"Mean Body MSE:       {np.mean(body_errs):.4f}" if body_errs else "Mean Body MSE:       N/A")
    print(f"Mean Left Hand MSE:  {np.mean(lhand_errs):.4f}" if lhand_errs else "Mean Left Hand MSE:  N/A")
    print(f"Mean Right Hand MSE: {np.mean(rhand_errs):.4f}" if rhand_errs else "Mean Right Hand MSE: N/A")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python evaluate_results.py <estimate_file> <openpose_file>")
        sys.exit(1)

    estimate_file = sys.argv[1]
    openpose_file = sys.argv[2]
    
    # Load the estimated and OpenPose keypoints
    estimate = np.load(estimate_file, allow_pickle=True) # Make sure to extract dictionary if saved using np.savez/dict
    openpose = np.load(openpose_file, allow_pickle=True)
    
    evaluate(estimate, openpose)