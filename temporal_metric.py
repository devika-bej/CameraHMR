import numpy as np
import sys
import torch

from CamSMPLifyX.utils.smplx_openpose import SMPLX_
from CamSMPLifyX.constants import SMPLX_MODEL_DIR, NUM_BETAS_SMPLX

def extract_3d_joints(estimate_file, device, model):
    """Loads an estimate file and extracts the 3D joint sequence."""
    estimate = np.load(estimate_file, allow_pickle=True)
    n_frames = estimate['imgname'].shape[0]
    
    joints_seq = []
    for i in range(n_frames):
        global_orient = torch.tensor(np.expand_dims(estimate["global_orient"][i], axis=0)).to(device).float()
        body_pose = torch.tensor(np.expand_dims(estimate["body_pose"][i], axis=0)).to(device).float()
        left_hand_pose = torch.tensor(np.expand_dims(estimate["left_hand_pose"][i], axis=0)).to(device).float()
        right_hand_pose = torch.tensor(np.expand_dims(estimate["right_hand_pose"][i], axis=0)).to(device).float()
        betas = torch.tensor(np.expand_dims(estimate["shape"][i], axis=0)).to(device).float()
        
        with torch.no_grad():
            smplx_output = model(
                global_orient=global_orient,
                body_pose=body_pose,
                left_hand_pose=left_hand_pose,
                right_hand_pose=right_hand_pose,
                betas=betas)
            joints_seq.append(smplx_output.joints.squeeze(0).cpu().numpy())
            
    return np.array(joints_seq) # Shape: (T, Num_Joints, 3)

def analyze_kinematics(cam_file, base_file, stitch_file):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SMPLX_(SMPLX_MODEL_DIR, num_betas=NUM_BETAS_SMPLX, use_pca=False).to(device)
    
    # Extract 3D joints for all three methods
    cam_joints = extract_3d_joints(cam_file, device, model)
    baseline_joints = extract_3d_joints(base_file, device, model)
    stitched_joints = extract_3d_joints(stitch_file, device, model)
    
    # SMPLX Joint ID 52 is roughly the Right Index Fingertip
    JOINT_ID = 52 
    AXIS = 2 # Z-axis (depth)
    
    # Extract 1D trajectories
    cam_traj = cam_joints[:, JOINT_ID, AXIS]
    base_traj = baseline_joints[:, JOINT_ID, AXIS]
    stitch_traj = stitched_joints[:, JOINT_ID, AXIS]
    
    # Calculate Acceleration (2nd derivative)
    cam_acc = np.diff(cam_traj, n=2)
    base_acc = np.diff(base_traj, n=2)
    stitch_acc = np.diff(stitch_traj, n=2)
    
    # Print metrics EXACTLY as expected by the bash script grep commands
    print(f"CameraHMR Mean Jitter (Abs Accel): {np.mean(np.abs(cam_acc)):.6f}")
    print(f"Baseline Mean Jitter (Abs Accel): {np.mean(np.abs(base_acc)):.6f}")
    print(f"Stitched Mean Jitter (Abs Accel): {np.mean(np.abs(stitch_acc)):.6f}")


if __name__ == "__main__":
    # Expecting 3 arguments plus the script name itself
    if len(sys.argv) != 4:
        print("Usage: python evaluate_temporal.py <camerhmr_estimate.npz> <baseline_estimate.npz> <stitched_estimate.npz>")
        sys.exit(1)
        
    cam_file = sys.argv[1]
    base_file = sys.argv[2]
    stitch_file = sys.argv[3]
    
    analyze_kinematics(cam_file, base_file, stitch_file)
