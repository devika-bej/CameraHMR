import numpy as np
import sys
import torch
import smplx
import os

from CamSMPLifyX.utils.smplx_openpose import SMPLX_
from CamSMPLifyX.constants import (
    SMPLX_MODEL_DIR,
    NUM_BETAS_SMPLX,
)

IMG_RES = 768

def get_transform(center, scale, res):
    """Generate affine transform matrix from original image to crop space."""
    h = 200 * scale
    t = torch.zeros(3, 3, device=center.device, dtype=center.dtype)

    t[0, 0] = res[1] / h[0]
    t[1, 1] = res[0] / h[1]
    t[0, 2] = res[1] * (-center[0].float() / h[0] + 0.5)
    t[1, 2] = res[0] * (-center[1].float() / h[1] + 0.5)
    t[2, 2] = 1.0

    return t

def transform(pts, center, scale, res):
    """Transform pixel locations into crop coordinate system."""
    t = get_transform(center, scale, res)
    ones_column = torch.ones(pts.shape[0], 1, device=pts.device, dtype=pts.dtype)

    pts_homo = torch.cat((pts, ones_column), dim=1)
    new_pts = torch.matmul(t, pts_homo.t()).t()
    new_pts = new_pts[:, :2] / new_pts[:, 2].unsqueeze(1)

    return new_pts + 1.0

def j2d_processing(kp, center, scale):
    """Project original-image 2D points into the 768x768 crop space."""
    kp_transformed = transform(kp + 1.0, center, scale, [IMG_RES, IMG_RES])
    return kp_transformed

def perspective_projection(points, translation, cam_intrinsics):
    """Project 3D points using camera translation and intrinsics."""
    K = cam_intrinsics
    points_translated = points + translation.unsqueeze(0)
    z = points_translated[:, 2].unsqueeze(-1).clamp(min=1e-6)

    projected_points = points_translated / z
    projected_points = torch.einsum("ij,kj->ki", K, projected_points.float())

    return projected_points

def evaluate(estimate, mp_data):
    n_frames = estimate['imgname'].shape[0]
    lhand_errs = []
    rhand_errs = []
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SMPLX_(SMPLX_MODEL_DIR, num_betas=NUM_BETAS_SMPLX, use_pca=False).to(device)
    
    for i in range(n_frames):
        # Target: Slice to [:, :2] to ensure we only grab X,Y from the MediaPipe file
        mp_rhand = mp_data['mediapipe_kp_left'][i][:, :2]
        mp_lhand = mp_data['mediapipe_kp_right'][i][:, :2]
        
        # Load SMPL-X Params & Camera info from the Estimate file
        global_orient = torch.tensor(np.expand_dims(estimate["global_orient"][i], axis=0)).to(device).float()
        cam_int_np = estimate["cam_int"][i]
        cam_t_np = estimate["cam_t"][i]
        center = torch.tensor(estimate["center"][i]).to(device).float()
        scale = torch.tensor(estimate["scale"][i]).to(device).float()

        body_pose = torch.tensor(np.expand_dims(estimate["body_pose"][i], axis=0)).to(device).float()
        left_hand_pose = torch.tensor(np.expand_dims(estimate["left_hand_pose"][i], axis=0)).to(device).float()
        right_hand_pose = torch.tensor(np.expand_dims(estimate["right_hand_pose"][i], axis=0)).to(device).float()
        betas = torch.tensor(np.expand_dims(estimate["shape"][i], axis=0)).to(device).float()
        
        c_int = torch.tensor(cam_int_np).unsqueeze(0).to(device).float()
        c_t = torch.tensor(cam_t_np).to(device).float()
        
        smplx_output = model(
            global_orient=global_orient,
            body_pose=body_pose,
            left_hand_pose=left_hand_pose,
            right_hand_pose=right_hand_pose,
            betas=betas
        )

        joints_3d = smplx_output.joints.squeeze(0)

        # MediaPipe & OpenPose share the identical 21-joint hand mapping array
        lhand_mapping = [20, 37, 38, 39, 66, 25, 26, 27, 67, 28, 29, 30, 68, 34, 35, 36, 69, 31, 32, 33, 70]
        rhand_mapping = [21, 52, 53, 54, 71, 40, 41, 42, 72, 43, 44, 45, 73, 49, 50, 51, 74, 46, 47, 48, 75]

        smplx_lhand_3d = joints_3d[lhand_mapping]
        smplx_rhand_3d = joints_3d[rhand_mapping]

        K = c_int.squeeze(0) 
        
        proj_lhand = perspective_projection(smplx_lhand_3d, c_t, K)
        proj_rhand = perspective_projection(smplx_rhand_3d, c_t, K)

        proj_lhand_2d = proj_lhand[:, :2]
        proj_rhand_2d = proj_rhand[:, :2]

        est_lhand_crop = j2d_processing(proj_lhand_2d, center, scale) / IMG_RES
        est_rhand_crop = j2d_processing(proj_rhand_2d, center, scale) / IMG_RES

        mp_lhand_t = torch.tensor(np.array(mp_lhand, dtype=np.float32)).to(device) / torch.tensor([IMG_RES, IMG_RES], device=device)
        mp_rhand_t = torch.tensor(np.array(mp_rhand, dtype=np.float32)).to(device) / torch.tensor([IMG_RES, IMG_RES], device=device)

        def compute_mse(est_2d, target_2d):
            """Calculates MSE for purely 2D coordinates (no confidence). Filters out (0,0) empty targets."""
            valid_mask = (target_2d[:, 0] != 0.0) & (target_2d[:, 1] != 0.0)
            
            if valid_mask.sum() == 0:
                return torch.tensor(0.0).to(device)
            
            err = (est_2d[valid_mask] - target_2d[valid_mask]) ** 2
            return err.sum(dim=-1).mean()

        mse_lhand = compute_mse(est_lhand_crop, mp_lhand_t)
        mse_rhand = compute_mse(est_rhand_crop, mp_rhand_t)

        lhand_errs.append(mse_lhand.item())
        rhand_errs.append(mse_rhand.item())

    print(f"Total Frames Evaluated:   {n_frames}")
    print(f"Mean L-Hand MSE: {np.mean(lhand_errs):.6f}")
    print(f"Mean R-Hand MSE: {np.mean(rhand_errs):.6f}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python mediapipe_metric.py <estimate_file> <mediapipe_file>")
        sys.exit(1)

    estimate_file = sys.argv[1]
    mediapipe_file = sys.argv[2]
    
    # Load the estimated and mediapipe keypoints separately
    estimate = np.load(estimate_file, allow_pickle=True)
    mp_data = np.load(mediapipe_file, allow_pickle=True)
    
    evaluate(estimate, mp_data)
