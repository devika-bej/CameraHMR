import torch
import sys
import smplx
import numpy as np
import os
import cv2

import torch.nn.functional as F

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

# This order assumes your MediaPipe keypoints have been reordered to:
# Wrist, Index, Middle, Ring, Pinky, Thumb
HAND_BONES = [
    # Index
    (0, 1), (1, 2), (2, 3), (3, 4),

    # Middle
    (0, 5), (5, 6), (6, 7), (7, 8),

    # Ring
    (0, 9), (9, 10), (10, 11), (11, 12),

    # Pinky
    (0, 13), (13, 14), (14, 15), (15, 16),

    # Thumb
    (0, 17), (17, 18), (18, 19), (19, 20),
]

IMG_RES = 768


def _safe_int_point(pt):
    """Convert a 2D point to int tuple if finite, otherwise return None."""
    if not np.isfinite(pt).all():
        return None
    return int(pt[0]), int(pt[1])


def _save_overlay(image_path, bbox_center, bbox_scale, mano_proj, mp_xy, out_path):
    """Draw MANO projections and MediaPipe targets on the cropped image."""
    img = cv2.imread(image_path)

    if img is None:
        print(f"  Could not load image from {image_path}")
        return

    img = crop(
        img,
        bbox_center.detach().cpu().numpy(),
        bbox_scale.detach().cpu().numpy(),
        [IMG_RES, IMG_RES]
    )

    img = np.clip(img, 0, 255).astype(np.uint8)
    img = np.ascontiguousarray(img)

    colors = {
        "mano_left": (255, 80, 80),      
        "mp_left": (80, 255, 80),        
        "mano_right": (255, 255, 80),    
        "mp_right": (80, 180, 255),      
        "line_left": (0, 0, 255),        
        "line_right": (255, 0, 255),     
    }

    for hand_idx, hand_name in enumerate(["left", "right"]):
        mano_hand = mano_proj[hand_idx]
        mp_hand = mp_xy[hand_idx]

        if hand_name == "left":
            mano_color = colors["mano_left"]
            mp_color = colors["mp_left"]
            line_color = colors["line_left"]
        else:
            mano_color = colors["mano_right"]
            mp_color = colors["mp_right"]
            line_color = colors["line_right"]

        for j in range(len(MANO_JOINT_NAMES)):
            mano_pt = _safe_int_point(mano_hand[j])
            mp_pt = _safe_int_point(mp_hand[j])

            if mano_pt is not None:
                cv2.circle(img, mano_pt, 6, mano_color, -1)

            if mp_pt is not None:
                cv2.circle(img, mp_pt, 6, mp_color, -1)

            if mano_pt is not None and mp_pt is not None:
                cv2.line(img, mano_pt, mp_pt, line_color, 1)

    cv2.putText(
        img,
        "Left: MANO blue, MP green",
        (20, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    cv2.putText(
        img,
        "Right: MANO cyan, MP orange",
        (20, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    cv2.imwrite(out_path, img)


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


def valid_keypoint_mask(kp_2d, eps=1e-6):
    """
    Returns a mask for valid 2D keypoints.
    """
    finite = torch.isfinite(kp_2d).all(dim=-1)
    non_zero = torch.linalg.norm(kp_2d, dim=-1) > eps
    return finite & non_zero


def masked_mean(values, mask):
    """
    Safe masked mean.
    """
    mask = mask.float()
    denom = mask.sum().clamp_min(1.0)
    return (values * mask).sum() / denom


def robust_2d_reprojection_loss(pred_2d, target_2d, valid_mask=None, delta=25.0):
    """
    Robust 2D reprojection loss using Huber penalty on joint-wise pixel error.
    """
    return torch.mean(F.huber_loss(pred_2d, target_2d))


def bone_direction_loss(pred_2d, target_2d, valid_mask=None, eps=1e-6):
    """
    2D bone direction loss.
    """
    device = pred_2d.device

    bones = torch.tensor(
        HAND_BONES,
        device=device,
        dtype=torch.long
    )

    start_idx = bones[:, 0]
    end_idx = bones[:, 1]

    pred_vec = pred_2d[..., end_idx, :] - pred_2d[..., start_idx, :]
    targ_vec = target_2d[..., end_idx, :] - target_2d[..., start_idx, :]
    
    return torch.mean(1 - F.cosine_similarity(pred_vec, targ_vec, eps=eps))


class HandOptimizer:
    def __init__(
        self,
        device=torch.device("cuda"),
        loss_weights=None
    ):
        self.device = device

        self.model = SMPLX_(
            SMPLX_MODEL_DIR,
            num_betas=NUM_BETAS_SMPLX,
            use_pca=False
        ).to(self.device)

        self.FINGER_TIPS_V_IDS_LH = [5361, 4933, 5058, 5169, 5286]
        self.FINGER_TIPS_V_IDS_RH = [8079, 7669, 7794, 7905, 8022]

        self.MP_TO_MANO_MAP = [
            100,
            12, 13, 14, -1,
            0, 1, 2, -2,
            3, 4, 5, -3,
            9, 10, 11, -4,
            6, 7, 8, -5
        ]

        self.loss_weights = {
            "kp2d": 1.0,
            "bone_dir": 150.0,
            "pose_prior": 500.0,  
            "wrist_prior": 50.0, # <-- New prior to prevent wrist breaking
        }

        if loss_weights is not None:
            self.loss_weights.update(loss_weights)


    def get_mano_landmarks(self, left_wrist, right_wrist, lh_joints, rh_joints, vertices):
        landmarks_lh = []
        landmarks_rh = []

        for idx in self.MP_TO_MANO_MAP:
            if idx == 100:
                landmarks_lh.append(left_wrist)
                landmarks_rh.append(right_wrist)

            elif idx >= 0:
                landmarks_lh.append(lh_joints[:, idx, :])
                landmarks_rh.append(rh_joints[:, idx, :])

            else:
                v_idx = abs(idx) - 1

                landmarks_lh.append(
                    vertices[:, self.FINGER_TIPS_V_IDS_LH[v_idx], :]
                )

                landmarks_rh.append(
                    vertices[:, self.FINGER_TIPS_V_IDS_RH[v_idx], :]
                )

        landmarks_lh = torch.stack(landmarks_lh, dim=1)
        landmarks_rh = torch.stack(landmarks_rh, dim=1)

        return landmarks_lh, landmarks_rh


    def compute_hand_losses(
        self,
        estimate_2d_lh,
        estimate_2d_rh,
        estimate_3d_lh,
        estimate_3d_rh,
        target_2d,
        target_3d,
        left_hand_pose,
        right_hand_pose,
        left_hand_pose_init,
        right_hand_pose_init,
        left_wrist,           # <-- Added wrist vars
        right_wrist,          
        left_wrist_init,      
        right_wrist_init      
    ):
        pred_lh = estimate_2d_lh[0]
        pred_rh = estimate_2d_rh[0]

        target_lh = target_2d[0]
        target_rh = target_2d[1]

        valid_lh = valid_keypoint_mask(target_lh)
        valid_rh = valid_keypoint_mask(target_rh)

        loss_2d_lh = robust_2d_reprojection_loss(
            pred_lh,
            target_lh,
            valid_mask=valid_lh,
            delta=25.0
        )

        loss_2d_rh = robust_2d_reprojection_loss(
            pred_rh,
            target_rh,
            valid_mask=valid_rh,
            delta=25.0
        )

        loss_2d = loss_2d_lh + loss_2d_rh

        loss_bone_lh = bone_direction_loss(
            pred_lh,
            target_lh,
            valid_mask=valid_lh
        )

        loss_bone_rh = bone_direction_loss(
            pred_rh,
            target_rh,
            valid_mask=valid_rh
        )

        loss_bone = loss_bone_lh + loss_bone_rh

        # --- THE CONCRETE HINGE PRIOR (Fingers) ---
        lh_joints = left_hand_pose.view(-1, 15, 3)
        rh_joints = right_hand_pose.view(-1, 15, 3)

        twist_yaw_penalty = torch.mean(lh_joints[:, :, 0:2] ** 2) + \
                            torch.mean(rh_joints[:, :, 0:2] ** 2)

        rh_backward_violation = F.relu(-rh_joints[:, :, 2]) 
        lh_backward_violation = F.relu(lh_joints[:, :, 2])  

        loss_hinge = torch.mean(rh_backward_violation ** 2) + \
                     torch.mean(lh_backward_violation ** 2)

        loss_pose_init = torch.mean((left_hand_pose - left_hand_pose_init) ** 2) + \
                         torch.mean((right_hand_pose - right_hand_pose_init) ** 2)

        total_pose_prior = (50.0 * loss_hinge) + (20.0 * twist_yaw_penalty) + (0.1 * loss_pose_init)
        
        # --- THE WRIST ANCHOR ---
        # Prevent the optimizer from twisting the wrist to compensate for rigid fingers
        loss_wrist = torch.mean((left_wrist - left_wrist_init) ** 2) + \
                     torch.mean((right_wrist - right_wrist_init) ** 2)
        # ------------------------

        total_loss = (
            self.loss_weights["kp2d"] * loss_2d
            + self.loss_weights["bone_dir"] * loss_bone
            + self.loss_weights["pose_prior"] * total_pose_prior
            + self.loss_weights["wrist_prior"] * loss_wrist # <-- Apply wrist penalty
        )

        loss_dict = {
            "total": total_loss,
            "kp2d": loss_2d.detach(),
            "bone_dir": loss_bone.detach(),
            "pose_prior": total_pose_prior.detach(),
            "wrist_prior": loss_wrist.detach(), # Add to log
        }

        return total_loss, loss_dict


    def refine(
        self,
        global_orient,
        body_pose,
        left_hand_pose,
        right_hand_pose,
        betas,
        cam_t,
        cam_int,
        bbox_center,
        bbox_scale,
        target_mp,
        img_path,
        num_epochs=300,
        lr=0.01,
        print_every=10,
        save_overlays=True
    ):

        global_orient = global_orient.to(self.device).float()
        body_pose = body_pose.to(self.device).float()
        betas = betas.to(self.device).float()
        cam_t = cam_t.to(self.device).float()
        cam_int = cam_int.to(self.device).float()
        bbox_center = bbox_center.to(self.device).float()
        bbox_scale = bbox_scale.to(self.device).float()
        target_mp = target_mp.to(self.device).float()

        left_hand_pose = (
            left_hand_pose
            .clone()
            .detach()
            .to(self.device)
            .float()
            .requires_grad_(True)
        )
        left_wrist = body_pose[:, 19].clone().requires_grad_(True)
        right_hand_pose = (
            right_hand_pose
            .clone()
            .detach()
            .to(self.device)
            .float()
            .requires_grad_(True)
        )
        right_wrist = body_pose[:, 20].clone().requires_grad_(True)

        left_hand_pose_init = left_hand_pose.clone().detach()
        right_hand_pose_init = right_hand_pose.clone().detach()
        
        # Save wrist initial states
        left_wrist_init = left_wrist.clone().detach()
        right_wrist_init = right_wrist.clone().detach()

        opt = torch.optim.Adam(
            [left_hand_pose, right_hand_pose, left_wrist, right_wrist],
            lr=lr
        )
        scheduler = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=[100, 200], gamma=0.1)
        
        bone_start, bone_end = 150.0, 5.0

        for epoch in range(num_epochs):
            progress = epoch / max(1, num_epochs - 1)
            
            self.loss_weights["bone_dir"] = bone_start * ((bone_end / bone_start) ** progress)

            opt.zero_grad()

            # --- CRITICAL FIX ---
            # Inject the optimized wrists back into the body_pose tensor 
            # so the model actually updates them physically in 3D space
            body_pose_opt = body_pose.clone()
            body_pose_opt[:, 19] = left_wrist
            body_pose_opt[:, 20] = right_wrist

            smplx_output = self.model(
                global_orient=global_orient,
                body_pose=body_pose_opt, # <-- Pass the updated tensor
                left_hand_pose=left_hand_pose,
                right_hand_pose=right_hand_pose,
                betas=betas
            )

            left_wrist_joint = smplx_output.joints[:, 20]
            lh_joints = smplx_output.joints[:, 25:40, :]
            right_wrist_joint = smplx_output.joints[:, 21]
            rh_joints = smplx_output.joints[:, 40:55, :]

            estimate_3d_lh, estimate_3d_rh = self.get_mano_landmarks(
                left_wrist_joint,
                right_wrist_joint,
                lh_joints,
                rh_joints,
                smplx_output.vertices
            )

            estimate_2d_lh = perspective_projection(
                estimate_3d_lh[0],
                cam_t,
                cam_int[0]
            )

            estimate_2d_lh = j2d_processing(
                estimate_2d_lh[:, :2],
                bbox_center,
                bbox_scale
            )

            estimate_2d_lh = estimate_2d_lh.unsqueeze(0)

            estimate_2d_rh = perspective_projection(
                estimate_3d_rh[0],
                cam_t,
                cam_int[0]
            )

            estimate_2d_rh = j2d_processing(
                estimate_2d_rh[:, :2],
                bbox_center,
                bbox_scale
            )

            estimate_2d_rh = estimate_2d_rh.unsqueeze(0)

            estimate_2d = torch.cat(
                (estimate_2d_lh, estimate_2d_rh),
                dim=0
            )

            target_2d = target_mp[:, :, :2]

            if save_overlays and epoch == 0:
                _save_overlay(
                    img_path,
                    bbox_center,
                    bbox_scale,
                    estimate_2d.detach().cpu().numpy(),
                    target_2d.detach().cpu().numpy(),
                    f"initial_overlay_{os.path.basename(img_path)}"
                )

            total_loss, loss_dict = self.compute_hand_losses(
                estimate_2d_lh=estimate_2d_lh,
                estimate_2d_rh=estimate_2d_rh,
                estimate_3d_lh=estimate_3d_lh,
                estimate_3d_rh=estimate_3d_rh,
                target_2d=target_2d,
                target_3d=target_mp,
                left_hand_pose=left_hand_pose,
                right_hand_pose=right_hand_pose,
                left_hand_pose_init=left_hand_pose_init,
                right_hand_pose_init=right_hand_pose_init,
                left_wrist=left_wrist,               # <-- Pass variables
                right_wrist=right_wrist,
                left_wrist_init=left_wrist_init,
                right_wrist_init=right_wrist_init
            )

            if (
                epoch % print_every == 0
                or epoch == num_epochs - 1
            ):
                current_lr = opt.param_groups[0]['lr']
                print(
                    f"Epoch {epoch + 1:04d}/{num_epochs} | "
                    f"LR: {current_lr:.4f} | "
                    f"total: {loss_dict['total'].item():.4f} | "
                    f"kp2d: {loss_dict['kp2d'].item():.4f} | "
                    f"prior: {loss_dict['pose_prior'].item():.4f} | " 
                    f"wrist: {loss_dict['wrist_prior'].item():.4f}" 
                )

            total_loss.backward()
            opt.step()
            scheduler.step()

        if save_overlays:
            with torch.no_grad():
                # Re-integrate for final output
                body_pose_opt = body_pose.clone()
                body_pose_opt[:, 19] = left_wrist
                body_pose_opt[:, 20] = right_wrist

                smplx_output = self.model(
                    global_orient=global_orient,
                    body_pose=body_pose_opt,
                    left_hand_pose=left_hand_pose,
                    right_hand_pose=right_hand_pose,
                    betas=betas
                )

                left_wrist_joint = smplx_output.joints[:, 20]
                lh_joints = smplx_output.joints[:, 25:40, :]
                right_wrist_joint = smplx_output.joints[:, 21]
                rh_joints = smplx_output.joints[:, 40:55, :]

                estimate_3d_lh, estimate_3d_rh = self.get_mano_landmarks(
                    left_wrist_joint,
                    right_wrist_joint,
                    lh_joints,
                    rh_joints,
                    smplx_output.vertices
                )

                estimate_2d_lh = perspective_projection(
                    estimate_3d_lh[0],
                    cam_t,
                    cam_int[0]
                )

                estimate_2d_lh = j2d_processing(
                    estimate_2d_lh[:, :2],
                    bbox_center,
                    bbox_scale
                ).unsqueeze(0)

                estimate_2d_rh = perspective_projection(
                    estimate_3d_rh[0],
                    cam_t,
                    cam_int[0]
                )

                estimate_2d_rh = j2d_processing(
                    estimate_2d_rh[:, :2],
                    bbox_center,
                    bbox_scale
                ).unsqueeze(0)

                estimate_2d = torch.cat(
                    (estimate_2d_lh, estimate_2d_rh),
                    dim=0
                )

                target_2d = target_mp[:, :, :2]

                _save_overlay(
                    img_path,
                    bbox_center,
                    bbox_scale,
                    estimate_2d.detach().cpu().numpy(),
                    target_2d.detach().cpu().numpy(),
                    f"final_overlay_{os.path.basename(img_path)}"
                )

        return left_hand_pose, right_hand_pose, left_wrist, right_wrist
    
    
    def smoothen(
        self, 
        left_hand_pose, 
        right_hand_pose,
        num_epochs=200,
        lr=0.01,
        w_data=1.0,
        w_vel=10.0,
        w_acc=100.0,
        print_every=20
    ):
        B = left_hand_pose.shape[0]
        if B < 3:
            print("Sequence too short for full temporal smoothing (needs B >= 3). Skipping.")
            return left_hand_pose, right_hand_pose

        left_init = left_hand_pose.clone().detach().to(self.device).float()
        right_init = right_hand_pose.clone().detach().to(self.device).float()

        left_opt = left_init.clone().requires_grad_(True)
        right_opt = right_init.clone().requires_grad_(True)

        optimizer = torch.optim.Adam([left_opt, right_opt], lr=lr)

        for epoch in range(num_epochs):
            optimizer.zero_grad()

            loss_data = torch.mean((left_opt - left_init) ** 2) + \
                        torch.mean((right_opt - right_init) ** 2)

            vel_left = left_opt[1:] - left_opt[:-1]
            vel_right = right_opt[1:] - right_opt[:-1]
            loss_vel = torch.mean(vel_left ** 2) + torch.mean(vel_right ** 2)

            acc_left = left_opt[2:] - 2 * left_opt[1:-1] + left_opt[:-2]
            acc_right = right_opt[2:] - 2 * right_opt[1:-1] + right_opt[:-2]
            loss_acc = torch.mean(acc_left ** 2) + torch.mean(acc_right ** 2)

            total_loss = (w_data * loss_data) + (w_vel * loss_vel) + (w_acc * loss_acc)

            total_loss.backward()
            optimizer.step()

            if (epoch % print_every == 0) or (epoch == num_epochs - 1):
                print(
                    f"Smoothen Epoch {epoch + 1:04d}/{num_epochs} | "
                    f"total: {total_loss.item():.4f} | "
                    f"data: {loss_data.item():.4f} | "
                    f"vel: {loss_vel.item():.4f} | "
                    f"acc: {loss_acc.item():.4f}"
                )
