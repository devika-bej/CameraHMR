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