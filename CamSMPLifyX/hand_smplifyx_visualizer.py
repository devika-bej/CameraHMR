import torch
import sys
import smplx
import numpy as np
import os
import cv2
from utils.image_utils import crop
from utils.smplx_openpose import SMPLX_
from constants import (
    SMPLX_MODEL_DIR,
    NUM_BETAS_SMPLX,
    SMPLX2SMPL,
    DOWNSAMPLE_MAT,
    LOSS_CUT,
    LOW_THRESHOLD,
    HIGH_THRESHOLD,
)

MANO_JOINT_NAMES = [
    "Wrist",
    "Index_MCP", "Index_PIP", "Index_DIP", "Index_TIP",
    "Middle_MCP", "Middle_PIP", "Middle_DIP", "Middle_TIP",
    "Ring_MCP", "Ring_PIP", "Ring_DIP", "Ring_TIP",
    "Pinky_MCP", "Pinky_PIP", "Pinky_DIP", "Pinky_TIP",
    "Thumb_MCP", "Thumb_PIP", "Thumb_DIP", "Thumb_TIP",
]

IMG_RES = 768

def get_transform(center, scale, res):
    """Generate transformation matrix."""
    h = 200 * scale
    t = torch.zeros(3, 3, device=center.device)  # Ensure device consistency
    t[0, 0] = res[1] / h[0]
    t[1, 1] = res[0]/ h[1]
    t[0, 2] = res[1] * (-center[0].float() / h[0] + .5)
    t[1, 2] = res[0] * (-center[1].float() / h[1] + .5)
    t[2, 2] = 1
    return t

def transform(pts, center, scale, res):
    """Transform pixel locations to a different reference."""
    t = get_transform(center, scale, res)
    ones_column = torch.ones(pts.shape[0], 1, device=pts.device)
    pts = torch.cat((pts, ones_column), dim=1)  # Add column of ones for homogeneous coordinates
    new_pts = torch.matmul(t, pts.t()).t()
    new_pts = new_pts[:, :2] / new_pts[:, 2].unsqueeze(1)  # Normalize homogeneous coordinates
    return new_pts + 1

def j2d_processing(kp, center, scale):
    kp_transformed = transform(kp + 1, center, scale, [IMG_RES, IMG_RES])
    # convert to normalized coordinates
    # kp[:, :-1] = 2.0 * kp[:, :-1] / IMG_RES - 1.0
    return kp_transformed

def perspective_projection(points, translation, cam_intrinsics):
    K = cam_intrinsics
    points_translated = points + translation.unsqueeze(0)
    projected_points = points_translated / points_translated[:, -1].unsqueeze(-1)
    projected_points = torch.einsum('ij,kj->ki', K, projected_points.float())
    return projected_points

class HandOptimizer:
    def __init__(self, device=torch.device("cuda")):
        self.device = device
        self.model = SMPLX_(SMPLX_MODEL_DIR, num_betas=NUM_BETAS_SMPLX, use_pca=False).to(self.device)
        self.FINGER_TIPS_V_IDS_LH = [5361, 4933, 5058, 5169, 5286]
        self.FINGER_TIPS_V_IDS_RH = [8079, 7669, 7794, 7905, 8022]
        self.MP_TO_MANO_MAP = [100, 12, 13, 14, -1, 0, 1, 2, -2, 3, 4, 5, -3, 9, 10, 11, -4, 6, 7, 8, -5]

    def get_mano_landmarks(self, body_joints, lh_joints, rh_joints, vertices):
        landmarks_lh = []
        landmarks_rh = []
        for idx in self.MP_TO_MANO_MAP:
            if idx == 100:
                landmarks_lh.append(body_joints[:, 20, :])
                landmarks_rh.append(body_joints[:, 21, :])
            elif idx >= 0:
                landmarks_lh.append(lh_joints[:, idx, :])
                landmarks_rh.append(rh_joints[:, idx, :])
            else:
                v_idx = abs(idx) - 1
                landmarks_lh.append(vertices[:, self.FINGER_TIPS_V_IDS_LH[v_idx], :])
                landmarks_rh.append(vertices[:, self.FINGER_TIPS_V_IDS_RH[v_idx], :])
        return torch.stack(landmarks_lh, dim=1), torch.stack(landmarks_rh, dim=1)

    def refine(self, global_orient, body_pose, left_hand_pose, right_hand_pose, betas, cam_t, cam_int, bbox_center, bbox_scale, target_mp, img_path):
        smplx_output = self.model(
            global_orient=global_orient,
            body_pose=body_pose,
            left_hand_pose=left_hand_pose,
            right_hand_pose=right_hand_pose,
            betas=betas
        )
        body_joints = smplx_output.joints
        lh_joints = smplx_output.joints[:, 25:40, :]
        rh_joints = smplx_output.joints[:, 40:55, :]
        
        estimate_3d_lh, estimate_3d_rh = self.get_mano_landmarks(body_joints, lh_joints, rh_joints, smplx_output.vertices)
        estimate_2d_lh = perspective_projection(estimate_3d_lh[0], cam_t, cam_int[0])
        estimate_2d_lh = j2d_processing(estimate_2d_lh[:, :-1], bbox_center, bbox_scale)
        estimate_2d_lh = estimate_2d_lh.unsqueeze(0)
        estimate_2d_rh = perspective_projection(estimate_3d_rh[0], cam_t, cam_int[0])
        estimate_2d_rh = j2d_processing(estimate_2d_rh[:, :-1], bbox_center, bbox_scale)
        estimate_2d_rh = estimate_2d_rh.unsqueeze(0)
        
        target_2d = target_mp[:, :, :2].to(self.device)
        
        img = cv2.imread(img_path)
        img_cropped = crop(img, bbox_center.cpu().numpy(), bbox_scale.cpu().numpy(), [IMG_RES, IMG_RES])
        for i in range(estimate_2d_lh.shape[1]):
            cv2.circle(img_cropped, (int(estimate_2d_lh[0, i, 0].item()), int(estimate_2d_lh[0, i, 1].item())), 5, (0, 255, 0), -1)
        for i in range(estimate_2d_rh.shape[1]):
            cv2.circle(img_cropped, (int(estimate_2d_rh[0, i, 0].item()), int(estimate_2d_rh[0, i, 1].item())), 5, (255, 0, 0), -1)
        for i in range(target_2d.shape[1]):
            cv2.circle(img_cropped, (int(target_2d[0, i, 0].item()), int(target_2d[0, i, 1].item())), 5, (0, 0, 255), -1)
        cv2.imwrite("hand_smplifyx_output.png", img_cropped)
        
if __name__ == "__main__":
    init_param_file = sys.argv[1]
    img_folder = sys.argv[2]
    inp_data = np.load(init_param_file, allow_pickle=True)
    mp_left = torch.tensor(np.expand_dims(inp_data["mediapipe_kp_left"][0], axis=0)).to("cuda").float()
    mp_right = torch.tensor(np.expand_dims(inp_data["mediapipe_kp_right"][0], axis=0)).to("cuda").float()
    cam_int = torch.tensor(np.expand_dims(inp_data["cam_int"][0], axis=0)).to("cuda").float()
    cam_t = torch.tensor(inp_data["cam_t"][0]).to("cuda").float()
    bbox_center = torch.tensor(inp_data["center"][0]).to("cuda").float()
    bbox_scale = torch.tensor(inp_data["scale"][0]).to("cuda").float()
    shape = torch.tensor(np.expand_dims(inp_data["shape"][0], axis=0)).to("cuda").float()
    pose = torch.tensor(np.expand_dims(inp_data["body_pose"][0], axis=0)).to("cuda").float()
    global_orient = torch.tensor(np.expand_dims(inp_data["global_orient"][0], axis=0)).to("cuda").float()
    lh_pose = torch.tensor(np.expand_dims(inp_data["left_hand_pose"][0], axis=0)).to("cuda").float()
    rh_pose = torch.tensor(np.expand_dims(inp_data["right_hand_pose"][0], axis=0)).to("cuda").float()
    
    optimizer = HandOptimizer()
    optimizer.refine(
        global_orient=global_orient,
        body_pose=pose,
        left_hand_pose=lh_pose,
        right_hand_pose=rh_pose,
        betas=shape,
        cam_t=cam_t,
        cam_int=cam_int,
        bbox_center=bbox_center,
        bbox_scale=bbox_scale,
        target_mp=torch.cat((mp_left, mp_right), dim=1),
        img_path=os.path.join(img_folder, inp_data["imgname"][0])
    )