import numpy as np
import sys
import torch
import os
import csv

from CamSMPLifyX.utils.smplx_openpose import SMPLX_
from CamSMPLifyX.constants import (
    SMPLX_MODEL_DIR,
    NUM_BETAS_SMPLX,
)

IMG_RES = 768

def get_transform(center, scale, res):
    h = 200 * scale
    t = torch.zeros(3, 3, device=center.device, dtype=center.dtype)
    t[0, 0] = res[1] / h[0]
    t[1, 1] = res[0] / h[1]
    t[0, 2] = res[1] * (-center[0].float() / h[0] + 0.5)
    t[1, 2] = res[0] * (-center[1].float() / h[1] + 0.5)
    t[2, 2] = 1.0
    return t

def transform(pts, center, scale, res):
    t = get_transform(center, scale, res)
    ones_column = torch.ones(pts.shape[0], 1, device=pts.device, dtype=pts.dtype)
    pts_homo = torch.cat((pts, ones_column), dim=1)
    new_pts = torch.matmul(t, pts_homo.t()).t()
    new_pts = new_pts[:, :2] / new_pts[:, 2].unsqueeze(1)
    return new_pts + 1.0

def j2d_processing(kp, center, scale):
    kp_transformed = transform(kp + 1.0, center, scale, [IMG_RES, IMG_RES])
    return kp_transformed

def perspective_projection(points, translation, cam_intrinsics):
    K = cam_intrinsics
    points_translated = points + translation.unsqueeze(0)
    z = points_translated[:, 2].unsqueeze(-1).clamp(min=1e-6)
    projected_points = points_translated / z
    projected_points = torch.einsum("ij,kj->ki", K, projected_points.float())
    return projected_points

def get_2d_joints_for_frame(estimate, frame_idx, model, device):
    """Helper function to extract 3D joints and project to 2D for a specific frame."""
    global_orient = torch.tensor(np.expand_dims(estimate["global_orient"][frame_idx], axis=0)).to(device).float()
    body_pose = torch.tensor(np.expand_dims(estimate["body_pose"][frame_idx], axis=0)).to(device).float()
    left_hand_pose = torch.tensor(np.expand_dims(estimate["left_hand_pose"][frame_idx], axis=0)).to(device).float()
    right_hand_pose = torch.tensor(np.expand_dims(estimate["right_hand_pose"][frame_idx], axis=0)).to(device).float()
    betas = torch.tensor(np.expand_dims(estimate["shape"][frame_idx], axis=0)).to(device).float()
    
    cam_int = torch.tensor(estimate["cam_int"][frame_idx]).unsqueeze(0).to(device).float()
    cam_t = torch.tensor(estimate["cam_t"][frame_idx]).to(device).float()
    center = torch.tensor(estimate["center"][frame_idx]).to(device).float()
    scale = torch.tensor(estimate["scale"][frame_idx]).to(device).float()
    
    smplx_output = model(
        global_orient=global_orient, body_pose=body_pose,
        left_hand_pose=left_hand_pose, right_hand_pose=right_hand_pose, betas=betas
    )

    joints_3d = smplx_output.joints.squeeze(0)

    lhand_mapping = [20, 37, 38, 39, 66, 25, 26, 27, 67, 28, 29, 30, 68, 34, 35, 36, 69, 31, 32, 33, 70]
    rhand_mapping = [21, 52, 53, 54, 71, 40, 41, 42, 72, 43, 44, 45, 73, 49, 50, 51, 74, 46, 47, 48, 75]

    proj_lhand = perspective_projection(joints_3d[lhand_mapping], cam_t, cam_int.squeeze(0))[:, :2]
    proj_rhand = perspective_projection(joints_3d[rhand_mapping], cam_t, cam_int.squeeze(0))[:, :2]

    est_lhand_crop = j2d_processing(proj_lhand, center, scale) / IMG_RES
    est_rhand_crop = j2d_processing(proj_rhand, center, scale) / IMG_RES
    
    return est_lhand_crop, est_rhand_crop

def evaluate_and_save(batch_name, sample_name, multi_est, mono_est, base_est, mp_data, output_csv):
    n_frames = multi_est['imgname'].shape[0]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SMPLX_(SMPLX_MODEL_DIR, num_betas=NUM_BETAS_SMPLX, use_pca=False).to(device)
    
    # Check if CSV exists to write the header
    file_exists = os.path.isfile(output_csv)
    
    csv_rows = []
    
    for i in range(n_frames):
        mp_rhand = mp_data['mediapipe_kp_left'][i][:, :2]
        mp_lhand = mp_data['mediapipe_kp_right'][i][:, :2]
        
        mp_lhand_t = torch.tensor(np.array(mp_lhand, dtype=np.float32)).to(device) / torch.tensor([IMG_RES, IMG_RES], device=device)
        mp_rhand_t = torch.tensor(np.array(mp_rhand, dtype=np.float32)).to(device) / torch.tensor([IMG_RES, IMG_RES], device=device)
        
        # Get 2D projections for all three methods
        multi_l, multi_r = get_2d_joints_for_frame(multi_est, i, model, device)
        mono_l, mono_r = get_2d_joints_for_frame(mono_est, i, model, device)
        base_l, base_r = get_2d_joints_for_frame(base_est, i, model, device)

        # Process Left Hand
        valid_mask_l = (mp_lhand_t[:, 0] != 0.0) & (mp_lhand_t[:, 1] != 0.0)
        valid_idxs_l = torch.where(valid_mask_l)[0]
        
        for idx in valid_idxs_l:
            d_multi = torch.norm(multi_l[idx] - mp_lhand_t[idx]).item()
            d_mono = torch.norm(mono_l[idx] - mp_lhand_t[idx]).item()
            d_base = torch.norm(base_l[idx] - mp_lhand_t[idx]).item()
            csv_rows.append([batch_name, sample_name, i, "LHand", idx.item(), d_multi, d_mono, d_base])

        # Process Right Hand
        valid_mask_r = (mp_rhand_t[:, 0] != 0.0) & (mp_rhand_t[:, 1] != 0.0)
        valid_idxs_r = torch.where(valid_mask_r)[0]
        
        for idx in valid_idxs_r:
            d_multi = torch.norm(multi_r[idx] - mp_rhand_t[idx]).item()
            d_mono = torch.norm(mono_r[idx] - mp_rhand_t[idx]).item()
            d_base = torch.norm(base_r[idx] - mp_rhand_t[idx]).item()
            csv_rows.append([batch_name, sample_name, i, "RHand", idx.item(), d_multi, d_mono, d_base])

    # Append to the master CSV file
    with open(output_csv, mode='a', newline='') as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(["Batch", "Sample", "Frame", "Body_Part", "Joint_Idx", "Dist_Multi", "Dist_Mono", "Dist_Base"])
        writer.writerows(csv_rows)
        
    print(f"Processed {sample_name} | Appended {len(csv_rows)} valid joint evaluations.")

if __name__ == "__main__":
    if len(sys.argv) != 8:
        print("Usage: python extract_global_distances.py <batch> <sample> <multi_file> <mono_file> <base_file> <mp_file> <output_csv>")
        sys.exit(1)

    batch_name = sys.argv[1]
    sample_name = sys.argv[2]
    
    multi_est = np.load(sys.argv[3], allow_pickle=True)
    mono_est = np.load(sys.argv[4], allow_pickle=True)
    base_est = np.load(sys.argv[5], allow_pickle=True)
    mp_data = np.load(sys.argv[6], allow_pickle=True)
    
    output_csv = sys.argv[7]
    
    evaluate_and_save(batch_name, sample_name, multi_est, mono_est, base_est, mp_data, output_csv)
