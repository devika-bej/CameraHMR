import numpy as np
import sys
import torch
import matplotlib.pyplot as plt
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

def analyze_kinematics(baseline_file, stitched_file):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SMPLX_(SMPLX_MODEL_DIR, num_betas=NUM_BETAS_SMPLX, use_pca=False).to(device)
    
    print("Extracting Baseline Joints...")
    baseline_joints = extract_3d_joints(baseline_file, device, model)
    print("Extracting Stitched Joints...")
    stitched_joints = extract_3d_joints(stitched_file, device, model)
    
    # SMPLX Joint ID 37 is roughly the Left Index Fingertip
    # SMPLX Joint ID 52 is roughly the Right Index Fingertip
    JOINT_ID = 52 
    AXIS = 2 # Z-axis (depth)
    
    # Extract 1D trajectories
    base_traj = baseline_joints[:, JOINT_ID, AXIS]
    stitch_traj = stitched_joints[:, JOINT_ID, AXIS]
    
    # Calculate Acceleration (2nd derivative)
    base_acc = np.diff(base_traj, n=2)
    stitch_acc = np.diff(stitch_traj, n=2)
    
    print(f"--- Kinematic Results ---")
    print(f"Baseline Mean Jitter (Abs Accel): {np.mean(np.abs(base_acc)):.6f}")
    print(f"Stitched Mean Jitter (Abs Accel): {np.mean(np.abs(stitch_acc)):.6f}")

    # Plot 1: 1D Trajectory
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(base_traj, label='Baseline (Monocular)', color='red', alpha=0.6)
    plt.plot(stitch_traj, label='Ours (Multi-view + Temporal)', color='blue', linewidth=2)
    plt.title("Right Index Fingertip Z-Trajectory")
    plt.xlabel("Frame")
    plt.ylabel("Z-Position")
    plt.legend()
    
    # Plot 2: Acceleration Histogram
    plt.subplot(1, 2, 2)
    plt.hist(base_acc, bins=30, alpha=0.5, color='red', label='Baseline')
    plt.hist(stitch_acc, bins=30, alpha=0.5, color='blue', label='Ours')
    plt.title("Inter-frame Acceleration Distribution")
    plt.xlabel("Acceleration Magnitude")
    plt.ylabel("Frequency")
    plt.legend()
    
    plt.tight_layout()
    plt.savefig("kinematic_analysis.png")
    print("Saved plots to kinematic_analysis.png")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python evaluate_temporal.py <baseline_estimate.npz> <stitched_estimate.npz>")
        sys.exit(1)
    analyze_kinematics(sys.argv[1], sys.argv[2])
