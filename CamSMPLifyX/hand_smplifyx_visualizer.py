import torch
import sys
import smplx
import numpy as np
import os
import cv2

MANO_JOINT_NAMES = [
    "Wrist",
    "Index_MCP", "Index_PIP", "Index_DIP", "Index_TIP",
    "Middle_MCP", "Middle_PIP", "Middle_DIP", "Middle_TIP",
    "Ring_MCP", "Ring_PIP", "Ring_DIP", "Ring_TIP",
    "Pinky_MCP", "Pinky_PIP", "Pinky_DIP", "Pinky_TIP",
    "Thumb_MCP", "Thumb_PIP", "Thumb_DIP", "Thumb_TIP",
]

IMG_RES = 768

def _save_overlay(image_path, mano_proj, mp_xy, out_path):
    """Draw both sets of landmarks on the image for visual inspection."""
    img = cv2.imread(image_path)
    if img is None:
        print(f"  Could not load image from {image_path}")
        return

    mano_proj = mano_proj[0]
    mp_xy = mp_xy[0]
    for i in range(len(MANO_JOINT_NAMES)):
        mx, my = int(mano_proj[i, 0]), int(mano_proj[i, 1])
        px, py = int(mp_xy[i, 0]),    int(mp_xy[i, 1])

        # MANO projected = blue circles
        cv2.circle(img, (mx, my), 6, (255, 80, 80), -1)
        # MediaPipe target = green circles
        cv2.circle(img, (px, py), 6, (80, 255, 80), -1)
        # Error line connecting them
        cv2.line(img, (mx, my), (px, py), (0, 0, 255), 1)

    # Legend
    cv2.putText(img, "MANO proj (blue)", (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 80, 80), 2)
    cv2.putText(img, "MediaPipe (green)", (20, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (80, 255, 80), 2)

    cv2.imwrite(out_path, img)

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
    def __init__(self, model_path, device=torch.device("cuda")):
        self.device = device
        self.models = {
            'left': smplx.create(model_path, model_type='mano', is_rhand=False, 
                                 use_pca=False, flat_hand_mean=True).to(device),
            'right': smplx.create(model_path, model_type='mano', is_rhand=True, 
                                  use_pca=False, flat_hand_mean=True).to(device)
        }
        
        self.FINGER_TIPS_V_IDS = [745, 317, 444, 556, 673]
        self.MP_TO_MANO_MAP = [0, 13, 14, 15, -1, 1, 2, 3, -2, 4, 5, 6, -3, 10, 11, 12, -4, 7, 8, 9, -5]

    def get_mano_landmarks(self, output):
        joints = output.joints 
        verts = output.vertices
        landmarks = []
        for idx in self.MP_TO_MANO_MAP:
            if idx >= 0:
                landmarks.append(joints[:, idx, :])
            else:
                v_idx = self.FINGER_TIPS_V_IDS[abs(idx) - 1]
                landmarks.append(verts[:, v_idx, :])
        return torch.stack(landmarks, dim=1)
    
    def refine(self, target_mp, init_pose, init_shape, cam_int, cam_t, bbox_center, bbox_scale, is_left=True, inp_image_path=None):
        side = 'left' if is_left else 'right'
        model = self.models[side]
        target_2d = target_mp[:, :, :2].to(self.device)
        
        init_pose = init_pose.reshape(1, -1)
        wrist_pose = init_pose[:, :3]
        hand_pose = init_pose[:, 3:]
        shape = init_shape
        bbox_center = bbox_center.to(self.device).float()
        bbox_scale = bbox_scale.to(self.device).float()

        output = model(hand_pose=hand_pose, global_orient=wrist_pose, betas=shape)
        estimate_3d = self.get_mano_landmarks(output), cam_t.to(self.device).float(), cam_int.to(self.device).float()
        # _save_overlay(inp_image_path, estimate_3d[0][0][:, :-1].unsqueeze(0).detach().cpu().numpy(), target_2d.detach().cpu().numpy(), f"initial_overlay_{side}.jpg")
        estimate_2d = perspective_projection(estimate_3d[0][0], cam_t.to(self.device).float(), cam_int.to(self.device).float()[0])
        estimate_2d = j2d_processing(estimate_2d[:, :-1], bbox_center, bbox_scale)
        estimate_2d = estimate_2d.unsqueeze(0)
        # estimate_2d = estimate_2d + bbox_center
        # estimate_2d = estimate_2d[:, :-1].unsqueeze(0)
        print(estimate_2d)
        print(target_2d)
        _save_overlay(inp_image_path, estimate_2d.detach().cpu().numpy(), target_2d.detach().cpu().numpy(), f"overlay_debug_{side}.jpg")


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
    
    lh_pose = torch.tensor(np.expand_dims(inp_data["left_hand_pose"][0], axis=0)).to("cuda").float()
    lh_pose = torch.cat([pose[:, 19:20, :], lh_pose], dim=1) # Combine Wrist and LH Pose
    rh_pose = torch.tensor(np.expand_dims(inp_data["right_hand_pose"][0], axis=0)).to("cuda").float()
    rh_pose = torch.cat([pose[:, 20:21, :], rh_pose], dim=1) # Combine Wrist and RH Pose

    # print("mp_left shape:", mp_left.shape)
    # print("mp_right shape:", mp_right.shape)
    # print("cam_int shape:", cam_int.shape)
    # print("cam_t shape:", cam_t.shape)
    # print("bbox_center shape:", bbox_center.shape)
    # print("bbox_scale shape:", bbox_scale.shape)
    # print("shape shape:", shape.shape)
    # print("pose shape:", pose.shape)
    # print("lh_pose shape:", lh_pose.shape)
    # print("rh_pose shape:", rh_pose.shape)

    hand_refiner = HandOptimizer(model_path="./data/models/")
    hand_refiner.refine(mp_left, lh_pose, shape[:, :10], cam_int, cam_t, bbox_center, bbox_scale, is_left=True, inp_image_path=os.path.join(img_folder, inp_data["imgname"][0]))
    hand_refiner.refine(mp_right, rh_pose, shape[:, :10], cam_int, cam_t, bbox_center, bbox_scale, is_left=False, inp_image_path=os.path.join(img_folder, inp_data["imgname"][0]))