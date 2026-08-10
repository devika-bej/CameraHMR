"""
Evaluate estimated SMPL-X hand poses focusing on 2D projection accuracy 
vs. 3D physical plausibility, temporal dynamics, and manifold stability.
"""

import os
import sys
import numpy as np
import torch

from CamSMPLifyX.utils.smplx_openpose import SMPLX_
from CamSMPLifyX.constants import SMPLX_MODEL_DIR, NUM_BETAS_SMPLX

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
IMG_RES = 768

HAND_KEY_FOR_LEFT = "mediapipe_kp_left"
HAND_KEY_FOR_RIGHT = "mediapipe_kp_right"

LHAND_JOINT_IDX = [20, 37, 38, 39, 66, 25, 26, 27, 67, 28, 29, 30, 68, 34, 35, 36, 69, 31, 32, 33, 70]
RHAND_JOINT_IDX = [21, 52, 53, 54, 71, 40, 41, 42, 72, 43, 44, 45, 73, 49, 50, 51, 74, 46, 47, 48, 75]


# ---------------------------------------------------------------------------
# Step 1: Frame Alignment
# ---------------------------------------------------------------------------
def align_frames(estimate, mp_data):
    if "imgname" not in estimate or "imgname" not in mp_data:
        raise KeyError("Both files must contain an 'imgname' field to align frames safely.")

    def normalize(n):
        return os.path.basename(n.decode("utf-8") if isinstance(n, bytes) else str(n))

    est_names = [normalize(n) for n in estimate["imgname"]]
    mp_names = [normalize(n) for n in mp_data["imgname"]]
    mp_lookup = {name: idx for idx, name in enumerate(mp_names)}

    est_idx, mp_idx = [], []
    for i, name in enumerate(est_names):
        j = mp_lookup.get(name)
        if j is not None:
            est_idx.append(i)
            mp_idx.append(j)

    if not est_idx:
        raise ValueError("No frames could be aligned between the two files.")

    return np.array(est_idx), np.array(mp_idx)


# ---------------------------------------------------------------------------
# Step 2: Batched SMPL-X Forward Pass
# ---------------------------------------------------------------------------
def run_smplx_batched(estimate, est_idx, model, device):
    """Note: model is now passed in as an argument."""
    N = len(est_idx)

    def batched(key):
        return torch.tensor(np.asarray(estimate[key])[est_idx]).float().to(device)

    num_expr = getattr(model, "num_expression_coeffs", 10)
    zeros = lambda dim: torch.zeros(N, dim, device=device, dtype=torch.float32)

    with torch.no_grad():
        output = model(
            global_orient=batched("global_orient"),
            body_pose=batched("body_pose"),
            left_hand_pose=batched("left_hand_pose"),
            right_hand_pose=batched("right_hand_pose"),
            betas=batched("shape"),
            jaw_pose=zeros(3),
            leye_pose=zeros(3),
            reye_pose=zeros(3),
            expression=zeros(num_expr),
        )
    return output.joints, output.vertices, model.faces_tensor


# ---------------------------------------------------------------------------
# Step 3: Camera Projection & Crop Space Mapping
# ---------------------------------------------------------------------------
def project_to_crop_space(joints_3d, estimate, est_idx, device):
    N, K, _ = joints_3d.shape

    cam_t = torch.tensor(np.asarray(estimate["cam_t"])[est_idx]).float().to(device)
    cam_int = torch.tensor(np.asarray(estimate["cam_int"])[est_idx]).float().to(device)
    center = torch.tensor(np.asarray(estimate["center"])[est_idx]).float().to(device)
    scale = torch.tensor(np.asarray(estimate["scale"])[est_idx]).float().to(device)

    if scale.dim() == 2 and scale.shape[1] == 1:
        scale = scale.squeeze(1)

    points = joints_3d + cam_t.unsqueeze(1)
    z = points[..., 2:3].clamp(min=1e-6)
    points = points / z
    projected = torch.einsum("nij,nkj->nki", cam_int, points)
    projected_2d = projected[..., :2]

    h = 200.0 * scale
    hx = hy = h if h.dim() == 1 else h[:, 0]
    hy = h if h.dim() == 1 else h[:, 1]

    tx = IMG_RES * (-center[:, 0] / hx + 0.5)
    ty = IMG_RES * (-center[:, 1] / hy + 0.5)
    sxy_x = (IMG_RES / hx)
    sxy_y = (IMG_RES / hy)

    pts = projected_2d + 1.0
    crop_x = pts[..., 0] * sxy_x.unsqueeze(1) + tx.unsqueeze(1)
    crop_y = pts[..., 1] * sxy_y.unsqueeze(1) + ty.unsqueeze(1)
    crop_xy = torch.stack([crop_x, crop_y], dim=-1) + 1.0

    return crop_xy / IMG_RES


# ---------------------------------------------------------------------------
# Step 4: Load Ground Truth Data
# ---------------------------------------------------------------------------
def load_mp_hand_2d(mp_data, key, mp_idx, device):
    raw = np.asarray(mp_data[key])[mp_idx].astype(np.float32)
    coords = raw[..., :2] / IMG_RES
    valid = ~((raw[..., 0] == 0.0) & (raw[..., 1] == 0.0))
    return torch.tensor(coords).to(device), torch.tensor(valid).to(device)


# ---------------------------------------------------------------------------
# Step 5: Metrics Implementation
# ---------------------------------------------------------------------------

def masked_mse_per_frame(est_xy, gt_xy, valid_mask):
    sq_err = ((est_xy - gt_xy) ** 2).sum(dim=-1)
    sq_err = torch.where(valid_mask, sq_err, torch.full_like(sq_err, float("nan")))
    with torch.no_grad():
        per_frame = torch.nanmean(sq_err, dim=1)
    return per_frame


def compute_pca_reconstruction_error(estimate, key, est_idx, model, num_comps=10, device='cuda'):
    """Measures how far the optimized pose has drifted from the MANO natural human distribution."""
    pose = torch.tensor(np.asarray(estimate[key])[est_idx]).float().to(device)
    
    if pose.shape[-1] != 45:
        return float('nan')

    if "left" in key:
        components = model.left_hand_components[:num_comps]  
        mean_pose = model.left_hand_mean                     
    else:
        components = model.right_hand_components[:num_comps] 
        mean_pose = model.right_hand_mean                    

    centered_pose = pose - mean_pose
    pca_weights = torch.matmul(centered_pose, components.T)          
    reconstructed_pose = torch.matmul(pca_weights, components)       
    
    reconstruction_error = torch.norm(centered_pose - reconstructed_pose, dim=-1) 
    return torch.mean(reconstruction_error).item()


def compute_temporal_jitter(joints_3d):
    """Measures the temporal smoothness (acceleration) of the 3D hand in mm/frame^2."""
    N, K, _ = joints_3d.shape
    if N < 3:
        return float('nan')

    acceleration = joints_3d[2:] - 2 * joints_3d[1:-1] + joints_3d[:-2]
    accel_mm = torch.norm(acceleration, dim=-1) * 1000.0
    return torch.mean(accel_mm).item()

def compute_biomechanical_synergy(estimate, key, est_idx, model, device='cuda'):
    """
    Measures the violation of DIP-PIP tendon coupling in the four main fingers.
    Dynamically converts PCA coefficients to 45D axis-angle rotations if required.
    """
    if key not in estimate:
        return float('nan')
        
    pose = torch.tensor(np.asarray(estimate[key])[est_idx]).float().to(device)
    
    # Clean any NaN values present in raw file
    if torch.isnan(pose).any():
        pose = torch.nan_to_num(pose, nan=0.0)

    # Reshape if stored as (N, 15, 3)
    if pose.dim() == 3:
        pose = pose.view(pose.shape[0], -1)

    N, dims = pose.shape

    # If pose is stored as PCA coefficients (e.g. 6D, 12D), reconstruct full 45D pose
    if dims != 45:
        if "left" in key:
            comps = model.left_hand_components[:dims]  # (dims, 45)
            mean = model.left_hand_mean                 # (45,)
        else:
            comps = model.right_hand_components[:dims] # (dims, 45)
            mean = model.right_hand_mean                # (45,)
            
        pose = torch.matmul(pose, comps) + mean        # Reconstructed (N, 45)
        
    pose_3d = pose.view(N, 15, 3)
    
    # MANO / SMPL-X Finger Joint Indices (PIP, DIP)
    # Index: (1, 2), Middle: (4, 5), Pinky: (7, 8), Ring: (10, 11)
    finger_pairs = [(1, 2), (4, 5), (7, 8), (10, 11)]
    
    total_violation = 0.0
    
    for pip_idx, dip_idx in finger_pairs:
        # Calculate flexion rotation magnitude (norm of axis-angle vector)
        pip_flexion = torch.norm(pose_3d[:, pip_idx, :], dim=-1)
        dip_flexion = torch.norm(pose_3d[:, dip_idx, :], dim=-1)
        
        # Anatomical tendon coupling: DIP flexes ~75% of PIP ($\theta_{DIP} \approx 0.75 \times \theta_{PIP}$)
        expected_dip = 0.75 * pip_flexion
        
        # Absolute error converted to degrees
        violation_deg = torch.abs(dip_flexion - expected_dip) * (180.0 / np.pi)
        total_violation += torch.nanmean(violation_deg).item()
        
    return total_violation / 4.0
def get_hand_pca_info(model, is_left, device):
    """
    Safely extracts PCA components and mean from the SMPL-X model,
    handling both Tensor and NumPy array formats.
    """
    if is_left:
        comps = getattr(model, 'left_hand_components', getattr(model, 'np_left_hand_components', None))
        mean = getattr(model, 'left_hand_mean', getattr(model, 'np_left_hand_mean', None))
    else:
        comps = getattr(model, 'right_hand_components', getattr(model, 'np_right_hand_components', None))
        mean = getattr(model, 'right_hand_mean', getattr(model, 'np_right_hand_mean', None))
        
    # Convert NumPy arrays to PyTorch Tensors if necessary
    if isinstance(comps, np.ndarray):
        comps = torch.tensor(comps)
    if isinstance(mean, np.ndarray):
        mean = torch.tensor(mean)
        
    return comps.float().to(device), mean.float().to(device)


def compute_pose_prior_energy(estimate, key, est_idx, model, device='cuda'):
    """
    Measures the Mahalanobis distance (Prior Energy) of the pose.
    Unconstrained optimizers will push this to massive values.
    """
    if key not in estimate:
        return float('nan')
        
    pose = torch.tensor(np.asarray(estimate[key])[est_idx]).float().to(device)
    pose = torch.nan_to_num(pose, nan=0.0)

    if pose.dim() == 3:
        pose = pose.view(pose.shape[0], -1)

    N, dims = pose.shape
    
    # If pose is saved as 45D axis-angle, project it down to latent PCA space
    if dims == 45: 
        comps, mean = get_hand_pca_info(model, "left" in key, device)
        # Get the top 10 PCA weights
        latent_code = torch.matmul(pose - mean, comps[:10].T)
    else:
        # If it's already saved as PCA coefficients, just use them directly
        latent_code = pose

    # The Energy is the squared L2 norm of the PCA weights
    energy = torch.sum(latent_code ** 2, dim=-1)
    return torch.mean(energy).item()


def compute_locked_distal_joints(estimate, key, est_idx, model, device='cuda'):
    """
    Flags the 'uncanny valley' stiff fingers seen in the unconstrained output.
    Counts how many PIP/DIP joints are unnaturally locked at < 3 degrees of flexion.
    """
    if key not in estimate:
        return float('nan')
        
    pose = torch.tensor(np.asarray(estimate[key])[est_idx]).float().to(device)
    pose = torch.nan_to_num(pose, nan=0.0)
    
    if pose.dim() == 3:
        pose = pose.view(pose.shape[0], -1)

    N, dims = pose.shape
    
    # Reconstruct to full 45D space if saved as PCA components
    if dims != 45: 
        comps, mean = get_hand_pca_info(model, "left" in key, device)
        pose = torch.matmul(pose, comps[:dims]) + mean

    pose_3d = pose.view(N, 15, 3)

    # Indices for PIP and DIP joints of the 4 main fingers
    # Index: (1, 2), Middle: (4, 5), Pinky: (7, 8), Ring: (10, 11)
    distal_joints = [1, 2, 4, 5, 7, 8, 10, 11]

    # Extract primary flexion axis magnitude (L2 norm of the axis-angle vector)
    flexion = torch.norm(pose_3d[:, distal_joints, :], dim=-1) # (N, 8)

    # Count how many joints are bent less than 3 degrees (approx 0.052 radians)
    locked_mask = flexion < 0.052
    locked_count = locked_mask.sum(dim=-1).float() # (N,)

    return torch.mean(locked_count).item()

# ---------------------------------------------------------------------------
# Entry Point & Reporting
# ---------------------------------------------------------------------------
def evaluate(estimate, mp_data):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    est_idx, mp_idx = align_frames(estimate, mp_data)
    n_frames = len(est_idx)

    # Initialize the model ONCE so it can be used for joints and PCA metrics
    model = SMPLX_(SMPLX_MODEL_DIR, num_betas=NUM_BETAS_SMPLX, use_pca=False).to(device)

    # Pass the instantiated model to the forward pass
    joints_3d, vertices, faces = run_smplx_batched(estimate, est_idx, model, device)
    
    lhand_3d = joints_3d[:, LHAND_JOINT_IDX, :]
    rhand_3d = joints_3d[:, RHAND_JOINT_IDX, :]

    est_lhand_2d = project_to_crop_space(lhand_3d, estimate, est_idx, device)
    est_rhand_2d = project_to_crop_space(rhand_3d, estimate, est_idx, device)

    gt_lhand_2d, gt_lhand_valid = load_mp_hand_2d(mp_data, HAND_KEY_FOR_LEFT, mp_idx, device)
    gt_rhand_2d, gt_rhand_valid = load_mp_hand_2d(mp_data, HAND_KEY_FOR_RIGHT, mp_idx, device)

    # --- Print Evaluation Summary ---
    print(f"==================================================")
    print(f"Total Frames Evaluated: {n_frames}")
    print(f"==================================================")

    # Baseline 2D Metric
    lhand_mse = masked_mse_per_frame(est_lhand_2d, gt_lhand_2d, gt_lhand_valid)
    rhand_mse = masked_mse_per_frame(est_rhand_2d, gt_rhand_2d, gt_rhand_valid)

    print(f"\n--- 2D MSE (Lower mathematically, but tracks projection trap) ---")
    if (~torch.isnan(lhand_mse)).sum() > 0:
        print(f"Mean L-Hand 2D MSE: {torch.nanmean(lhand_mse).item():.6f}")
    if (~torch.isnan(rhand_mse)).sum() > 0:
        print(f"Mean R-Hand 2D MSE: {torch.nanmean(rhand_mse).item():.6f}")

    if "left_hand_pose" in estimate:
        l_synergy = compute_biomechanical_synergy(estimate, "left_hand_pose", est_idx, device)
        print(f"Mean L-Hand Tendon Violation: {l_synergy:.2f}° per finger")
        
    if "right_hand_pose" in estimate:
        r_synergy = compute_biomechanical_synergy(estimate, "right_hand_pose", est_idx, device)
        print(f"Mean R-Hand Tendon Violation: {r_synergy:.2f}° per finger")

    print(f"\n--- Probabilistic & Structural Plausibility ---")
    
    if "left_hand_pose" in estimate:
        l_energy = compute_pose_prior_energy(estimate, "left_hand_pose", est_idx, model, device)
        l_locked = compute_locked_distal_joints(estimate, "left_hand_pose", est_idx, model, device)
        print(f"Mean L-Hand Prior Energy:      {l_energy:.2f}")
        print(f"Mean L-Hand Stiff/Locked Joints: {l_locked:.2f} out of 8 per frame")
        
    if "right_hand_pose" in estimate:
        r_energy = compute_pose_prior_energy(estimate, "right_hand_pose", est_idx, model, device)
        r_locked = compute_locked_distal_joints(estimate, "right_hand_pose", est_idx, model, device)
        print(f"Mean R-Hand Prior Energy:      {r_energy:.2f}")
        print(f"Mean R-Hand Stiff/Locked Joints: {r_locked:.2f} out of 8 per frame")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python evaluate_results.py <estimate_file.npz> <mediapipe_file.npz>")
        sys.exit(1)

    estimate = np.load(sys.argv[1], allow_pickle=True)
    mp_data = np.load(sys.argv[2], allow_pickle=True)

    evaluate(estimate, mp_data)
