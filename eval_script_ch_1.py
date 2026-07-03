import numpy as np
import sys
import torch
import smplx
import os
import cv2

from CamSMPLifyX.utils.smplx_openpose import SMPLX_
from CamSMPLifyX.constants import (
    SMPLX_MODEL_DIR,
    NUM_BETAS_SMPLX,
)

# Added IMG_RES variable as expected by j2d_processing
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

    ones_column = torch.ones(
        pts.shape[0],
        1,
        device=pts.device,
        dtype=pts.dtype
    )

    pts_homo = torch.cat((pts, ones_column), dim=1)
    new_pts = torch.matmul(t, pts_homo.t()).t()

    new_pts = new_pts[:, :2] / new_pts[:, 2].unsqueeze(1)

    return new_pts + 1.0


def j2d_processing(kp, center, scale):
    """Project original-image 2D points into the 768x768 crop space."""
    kp_transformed = transform(kp + 1.0, center, scale, [IMG_RES, IMG_RES])
    return kp_transformed


def perspective_projection(points, translation, cam_intrinsics):
    """
    Project 3D points using camera translation and intrinsics.
    """
    K = cam_intrinsics

    points_translated = points + translation.unsqueeze(0)

    z = points_translated[:, 2].unsqueeze(-1).clamp(min=1e-6)

    projected_points = points_translated / z
    projected_points = torch.einsum(
        "ij,kj->ki",
        K,
        projected_points.float()
    )

    return projected_points

def evaluate(estimate, openpose):
    n_frames = estimate['imgname'].shape[0]
    body_errs = []
    lhand_errs = []
    rhand_errs = []
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SMPLX_(SMPLX_MODEL_DIR, num_betas=NUM_BETAS_SMPLX, use_pca=False).to(device)
    
    for i in range(n_frames):
        # OpenPose targets: shape is usually (N, 3) where columns are [x, y, confidence]
        op_body = openpose['body_pose'][i]
        op_lhand = openpose['left_hand_pose'][i]
        op_rhand = openpose['right_hand_pose'][i]
        
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
            betas=betas)

        # 1. Map SMPLX joints to OpenPose Landmarks
        # Typically, a SMPLX wrapper customized for OpenPose outputs 118 or 137 joints.
        # Assuming the standard 118-joint mapping convention:
        # Body: 0-24, Face: 25-75, Left Hand: 76-96, Right Hand: 97-117
        joints_3d = smplx_output.joints.squeeze(0)  # Shape: (Num_Joints, 3)

        body_mapping = [55,12, 17, 19, 21, 16, 18, 20, 0, 2, 5, 8, 1, 4, 7, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65]
        lhand_mapping = [20, 37, 38, 39, 66, 25, 26, 27, 67, 28, 29, 30, 68, 34, 35, 36, 69, 31, 32, 33, 70]
        rhand_mapping = [21, 52, 53, 54, 71, 40, 41, 42, 72, 43, 44, 45, 73, 49, 50, 51, 74, 46, 47, 48, 75]

        
        smplx_body_3d = joints_3d[body_mapping]
        smplx_lhand_3d = joints_3d[lhand_mapping]
        smplx_rhand_3d = joints_3d[rhand_mapping]

        # 2. Use perspective projection to project those points into 2D camera coordinates
        # Squeeze c_int to [3, 3] as expected by the perspective_projection einsum operation
        K = c_int.squeeze(0) 
        
        proj_body = perspective_projection(smplx_body_3d, c_t, K)
        proj_lhand = perspective_projection(smplx_lhand_3d, c_t, K)
        proj_rhand = perspective_projection(smplx_rhand_3d, c_t, K)

        # Keep only X, Y (drop Z from projected)
        proj_body_2d = proj_body[:, :2]
        proj_lhand_2d = proj_lhand[:, :2]
        proj_rhand_2d = proj_rhand[:, :2]

        # 3. Use j2d_processing to get them in correct crop pixel space
        est_body_crop = j2d_processing(proj_body_2d, center, scale) / 768
        est_lhand_crop = j2d_processing(proj_lhand_2d, center, scale) / 768
        est_rhand_crop = j2d_processing(proj_rhand_2d, center, scale) / 768

        # 4. Calculate MSE between Estimate and Target
        # Cast the object arrays to float32 before converting to tensors
        op_body_t = torch.tensor(np.array(op_body, dtype=np.float32)).to(device) / torch.tensor([768, 768, 1.0], device=device)
        op_lhand_t = torch.tensor(np.array(op_lhand, dtype=np.float32)).to(device) / torch.tensor([768, 768, 1.0], device=device)
        op_rhand_t = torch.tensor(np.array(op_rhand, dtype=np.float32)).to(device) / torch.tensor([768, 768, 1.0], device=device)

        def compute_mse(est_2d, target_op):
            """Calculates MSE considering only joints with > 0.0 confidence"""
            conf = target_op[:, 2]
            valid_mask = conf > 0.0
            
            if valid_mask.sum() == 0:
                return torch.tensor(0.0).to(device)
            
            # Calculate (Estimate - Target)^2
            err = (est_2d[valid_mask] - target_op[valid_mask, :2]) ** 2
            # Sum over x,y coordinates and compute the mean over valid joints
            return err.sum(dim=-1).mean()

        mse_body = compute_mse(est_body_crop, op_body_t)
        mse_lhand = compute_mse(est_lhand_crop, op_lhand_t)
        mse_rhand = compute_mse(est_rhand_crop, op_rhand_t)

        body_errs.append(mse_body.item())
        lhand_errs.append(mse_lhand.item())
        rhand_errs.append(mse_rhand.item())

    # Final Summary Results
    print(f"Total Frames Evaluated: {n_frames}")
    print(f"Mean Body MSE:       {np.mean(body_errs):.4f}")
    print(f"Mean Left Hand MSE:  {np.mean(lhand_errs):.4f}")
    print(f"Mean Right Hand MSE: {np.mean(rhand_errs):.4f}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python evaluate_results.py <estimate_file> <openpose_file>")
        sys.exit(1)

    estimate_file = sys.argv[1]
    openpose_file = sys.argv[2]
    
    # Load the estimated and OpenPose keypoints
    estimate = np.load(estimate_file, allow_pickle=True)
    openpose = np.load(openpose_file, allow_pickle=True)
    
    evaluate(estimate, openpose)
