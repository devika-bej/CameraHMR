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

    # OpenPose Mappings
    body_mapping = [55,12, 17, 19, 21, 16, 18, 20, 0, 2, 5, 8, 1, 4, 7, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65]
    lhand_mapping = [20, 37, 38, 39, 66, 25, 26, 27, 67, 28, 29, 30, 68, 34, 35, 36, 69, 31, 32, 33, 70]
    rhand_mapping = [21, 52, 53, 54, 71, 40, 41, 42, 72, 43, 44, 45, 73, 49, 50, 51, 74, 46, 47, 48, 75]

    K = cam_int.squeeze(0)
    
    proj_body = perspective_projection(joints_3d[body_mapping], cam_t, K)[:, :2]
    proj_lhand = perspective_projection(joints_3d[lhand_mapping], cam_t, K)[:, :2]
    proj_rhand = perspective_projection(joints_3d[rhand_mapping], cam_t, K)[:, :2]

    est_body_crop = j2d_processing(proj_body, center, scale) / IMG_RES
    est_lhand_crop = j2d_processing(proj_lhand, center, scale) / IMG_RES
    est_rhand_crop = j2d_processing(proj_rhand, center, scale) / IMG_RES
    
    return est_body_crop, est_lhand_crop, est_rhand_crop

def evaluate_and_save(batch_name, sample_name, multi_est, mono_est, base_est, op_data, output_csv):
    n_frames = multi_est['imgname'].shape[0]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SMPLX_(SMPLX_MODEL_DIR, num_betas=NUM_BETAS_SMPLX, use_pca=False).to(device)
    
    file_exists = os.path.isfile(output_csv)
    csv_rows = []
    
    for i in range(n_frames):
        # Load OpenPose targets
        op_body = op_data['body_pose'][i]
        op_lhand = op_data['left_hand_pose'][i]
        op_rhand = op_data['right_hand_pose'][i]
        
        # Convert to tensors and normalize X, Y by 768. Confidence remains at scale 0-1.
        op_body_t = torch.tensor(np.array(op_body, dtype=np.float32)).to(device)
        op_body_t[:, :2] = op_body_t[:, :2] / IMG_RES
        
        op_lhand_t = torch.tensor(np.array(op_lhand, dtype=np.float32)).to(device)
        op_lhand_t[:, :2] = op_lhand_t[:, :2] / IMG_RES
        
        op_rhand_t = torch.tensor(np.array(op_rhand, dtype=np.float32)).to(device)
        op_rhand_t[:, :2] = op_rhand_t[:, :2] / IMG_RES
        
        # Get 2D projections for all three methods
        multi_b, multi_l, multi_r = get_2d_joints_for_frame(multi_est, i, model, device)
        mono_b, mono_l, mono_r = get_2d_joints_for_frame(mono_est, i, model, device)
        base_b, base_l, base_r = get_2d_joints_for_frame(base_est, i, model, device)

        def extract_dists(part_name, target_op, est_multi, est_mono, est_base):
            conf = target_op[:, 2]
            # Only process joints where OpenPose actually made a detection (conf > 0)
            valid_idxs = torch.where(conf > 0.0)[0]
            
            for idx in valid_idxs:
                c = conf[idx].item()
                # Euclidean distance using just the X, Y coordinates
                d_multi = torch.norm(est_multi[idx] - target_op[idx, :2]).item()
                d_mono = torch.norm(est_mono[idx] - target_op[idx, :2]).item()
                d_base = torch.norm(est_base[idx] - target_op[idx, :2]).item()
                
                # Notice we are now writing the Confidence 'c' into the CSV as well
                csv_rows.append([batch_name, sample_name, i, part_name, idx.item(), c, d_multi, d_mono, d_base])

        # Extract for all three body parts
        extract_dists("Body", op_body_t, multi_b, mono_b, base_b)
        extract_dists("LHand", op_lhand_t, multi_l, mono_l, base_l)
        extract_dists("RHand", op_rhand_t, multi_r, mono_r, base_r)

    # Append to the master CSV file
    with open(output_csv, mode='a', newline='') as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(["Batch", "Sample", "Frame", "Body_Part", "Joint_Idx", "Confidence", "Dist_Multi", "Dist_Mono", "Dist_Base"])
        writer.writerows(csv_rows)
        
    print(f"Processed {sample_name} | Appended {len(csv_rows)} valid joint evaluations.")

if __name__ == "__main__":
    if len(sys.argv) != 8:
        print("Usage: python extract_global_distances_op.py <batch> <sample> <multi_file> <mono_file> <base_file> <openpose_file> <output_csv>")
        sys.exit(1)

    batch_name = sys.argv[1]
    sample_name = sys.argv[2]
    
    multi_est = np.load(sys.argv[3], allow_pickle=True)
    mono_est = np.load(sys.argv[4], allow_pickle=True)
    base_est = np.load(sys.argv[5], allow_pickle=True)
    op_data = np.load(sys.argv[6], allow_pickle=True)
    
    output_csv = sys.argv[7]
    
    evaluate_and_save(batch_name, sample_name, multi_est, mono_est, base_est, op_data, output_csv)
