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

    # mano_proj shape: [2, 21, 2]
    # mp_xy shape:    [2, 21, 2]
    # 0 = left, 1 = right

    colors = {
        "mano_left": (255, 80, 80),      # blue-ish
        "mp_left": (80, 255, 80),        # green
        "mano_right": (255, 255, 80),    # cyan-ish
        "mp_right": (80, 180, 255),      # orange-ish
        "line_left": (0, 0, 255),        # red
        "line_right": (255, 0, 255),     # magenta
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

    points:         [N, 3]
    translation:    [3]
    cam_intrinsics: [3, 3]

    returns:        [N, 3], where first two columns are pixel x/y.
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

    kp_2d: [21, 2]

    This masks NaNs/Infs and all-zero missing detections.
    """
    finite = torch.isfinite(kp_2d).all(dim=-1)
    non_zero = torch.linalg.norm(kp_2d, dim=-1) > eps
    return finite & non_zero


def masked_mean(values, mask):
    """
    Safe masked mean.

    values: [N]
    mask:   [N]
    """
    mask = mask.float()
    denom = mask.sum().clamp_min(1.0)
    return (values * mask).sum() / denom


def robust_2d_reprojection_loss(pred_2d, target_2d, valid_mask=None, delta=25.0):
    """
    Robust 2D reprojection loss using Huber penalty on joint-wise pixel error.

    pred_2d:   [21, 2]
    target_2d: [21, 2]

    Returns a scalar.
    """
    diff = pred_2d - target_2d
    err = torch.sqrt(torch.sum(diff ** 2, dim=-1) + 1e-8)

    # Huber on Euclidean pixel error.
    # For small errors: quadratic.
    # For large errors: linear, so noisy MediaPipe points do not dominate.
    quadratic = 0.5 * (err ** 2) / delta
    linear = err - 0.5 * delta

    loss_per_joint = torch.where(err < delta, quadratic, linear)

    if valid_mask is None:
        return loss_per_joint.mean()

    return masked_mean(loss_per_joint, valid_mask)


def bone_direction_loss(pred_2d, target_2d, valid_mask=None, eps=1e-6):
    """
    2D bone direction loss.

    Encourages each predicted finger bone to point in the same 2D direction
    as the target bone.

    This is more useful than 2D bone length loss because bone lengths can vary
    with crop scale, perspective, and noisy detections.

    pred_2d:   [21, 2]
    target_2d: [21, 2]
    """
    device = pred_2d.device

    bones = torch.tensor(
        HAND_BONES,
        device=device,
        dtype=torch.long
    )

    start_idx = bones[:, 0]
    end_idx = bones[:, 1]

    pred_vec = pred_2d[end_idx] - pred_2d[start_idx]
    targ_vec = target_2d[end_idx] - target_2d[start_idx]

    pred_len = torch.linalg.norm(pred_vec, dim=-1).clamp_min(eps)
    targ_len = torch.linalg.norm(targ_vec, dim=-1).clamp_min(eps)

    pred_dir = pred_vec / pred_len.unsqueeze(-1)
    targ_dir = targ_vec / targ_len.unsqueeze(-1)

    cosine = torch.sum(pred_dir * targ_dir, dim=-1)
    cosine = cosine.clamp(min=-1.0, max=1.0)

    loss_per_bone = 1.0 - cosine

    bone_valid = torch.ones(
        len(HAND_BONES),
        device=device,
        dtype=torch.bool
    )

    if valid_mask is not None:
        bone_valid = valid_mask[start_idx] & valid_mask[end_idx]

    return masked_mean(loss_per_bone, bone_valid)


def pose_deviation_prior(current_pose, initial_pose):
    """
    Penalize moving too far from the initial hand pose.

    This is useful for single-image fitting because 2D keypoints alone are
    ambiguous and can push the hand into implausible 3D poses.
    """
    return torch.mean((current_pose - initial_pose) ** 2)


def axis_angle_to_matrix(axis_angle):
    """
    Convert axis-angle representation to a 3x3 rotation matrix.
    axis_angle: Tensor of shape (..., 3)
    Returns: Tensor of shape (..., 3, 3)
    """
    # Calculate the angle (magnitude) and the normalized axis
    angle = torch.linalg.norm(axis_angle, dim=-1, keepdim=True)
    axis = axis_angle / (angle + 1e-7)  # Add epsilon to prevent division by zero
    
    cos_a = torch.cos(angle)
    sin_a = torch.sin(angle)
    
    # Split axis components
    x, y, z = axis[..., 0:1], axis[..., 1:2], axis[..., 2:3]
    
    # Create the skew-symmetric matrix K
    zero = torch.zeros_like(x)
    K = torch.stack([
        torch.cat([zero, -z, y], dim=-1),
        torch.cat([z, zero, -x], dim=-1),
        torch.cat([-y, x, zero], dim=-1)
    ], dim=-2)
    
    # Identity matrix
    I = torch.eye(3, device=axis_angle.device, dtype=axis_angle.dtype)
    
    # Rodrigues' formula: R = I + sin(a)*K + (1 - cos(a))*K^2
    R = I + sin_a.unsqueeze(-1) * K + (1 - cos_a.unsqueeze(-1)) * torch.matmul(K, K)
    return R


def matrix_to_euler_angles(matrix):
    """
    Convert a 3x3 rotation matrix to XYZ Euler angles.
    matrix: Tensor of shape (..., 3, 3)
    Returns: Tensor of shape (..., 3) containing (x, y, z) angles in radians.
    """
    # Calculate cosine of the y-angle (pitch) to check for Gimbal lock
    sy = torch.sqrt(matrix[..., 0, 0]**2 + matrix[..., 0, 1]**2)
    
    # Boolean mask for Gimbal lock condition
    singular = sy < 1e-6
    
    # Normal case
    x = torch.atan2(-matrix[..., 1, 2], matrix[..., 2, 2])
    y = torch.atan2(matrix[..., 0, 2], sy)
    z = torch.atan2(-matrix[..., 0, 1], matrix[..., 0, 0])
    
    # Singular case (Gimbal lock)
    x_sing = torch.atan2(matrix[..., 2, 1], matrix[..., 1, 1])
    y_sing = torch.atan2(matrix[..., 0, 2], sy)
    z_sing = torch.zeros_like(x_sing)
    
    # Use torch.where to apply the singular case math only where needed
    x = torch.where(singular, x_sing, x)
    y = torch.where(singular, y_sing, y)
    z = torch.where(singular, z_sing, z)
    
    return torch.stack([x, y, z], dim=-1)


def axis_angle_limit_loss(hand_pose, max_angle=1.5):
    """
    Soft limit on axis-angle magnitudes.

    hand_pose is usually [1, 45] for 15 hand joints x 3 axis-angle values.
    This is a simple anatomical regularizer to discourage extreme rotations.
    """
    # hand_pose shape: [batch, 45] (15 joints x 3)
    pose = hand_pose.reshape(-1, 3)
    
    # 1. Convert axis-angle to Rotation Matrices, then to Euler Angles
    rot_mats = axis_angle_to_matrix(pose)
    # Using 'XYZ' convention. You must check which axis your specific hand model 
    # (like MANO) uses for flexion/extension. Let's assume it's X.
    euler_angles = matrix_to_euler_angles(rot_mats) 
    
    # 2. Define anatomical limits (in radians) for each local axis
    # X-axis (Flexion/Extension): Allow large forward bend, heavily penalize backward bend
    x_min, x_max = -0.1, 1.6  # approx -5 deg to +90 deg
    
    # Y and Z axes (Abduction/Twisting): Fingers don't twist or spread much
    yz_limit = 0.2 # approx 11 degrees
    
    # 3. Calculate excess for each axis independently
    x_excess_upper = torch.relu(euler_angles[:, 0] - x_max)
    x_excess_lower = torch.relu(x_min - euler_angles[:, 0]) # Penalizes going below x_min
    
    y_excess = torch.relu(torch.abs(euler_angles[:, 1]) - yz_limit)
    z_excess = torch.relu(torch.abs(euler_angles[:, 2]) - yz_limit)
    
    # 4. Combine all violations into a single loss
    total_excess = x_excess_upper + x_excess_lower + y_excess + z_excess
    
    return torch.mean(total_excess ** 2)


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

        # Mapping from your 21-point hand order:
        # Wrist, Index, Middle, Ring, Pinky, Thumb
        #
        # 100 = wrist from body joints
        # non-negative = SMPL-X hand joint index
        # negative = fingertip vertex index
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
            "bone_dir": 80.0,
            "pose_prior": 0.05,
            "angle_limit": 2.0,
        }

        if loss_weights is not None:
            self.loss_weights.update(loss_weights)

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
        target_2d,
        left_hand_pose,
        right_hand_pose,
        left_hand_pose_init,
        right_hand_pose_init
    ):
        """
        Computes all single-image hand fitting losses.

        estimate_2d_lh: [1, 21, 2]
        estimate_2d_rh: [1, 21, 2]
        target_2d:      [2, 21, 2]
        """
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

        loss_pose = (
            pose_deviation_prior(left_hand_pose, left_hand_pose_init)
            + pose_deviation_prior(right_hand_pose, right_hand_pose_init)
        )

        loss_angle = (
            axis_angle_limit_loss(left_hand_pose, max_angle=1)
            + axis_angle_limit_loss(right_hand_pose, max_angle=1)
        )

        total_loss = (
            self.loss_weights["kp2d"] * loss_2d
            + self.loss_weights["bone_dir"] * loss_bone
            + self.loss_weights["pose_prior"] * loss_pose
            + self.loss_weights["angle_limit"] * loss_angle
        )

        loss_dict = {
            "total": total_loss,
            "kp2d": loss_2d.detach(),
            "bone_dir": loss_bone.detach(),
            "pose_prior": loss_pose.detach(),
            "angle_limit": loss_angle.detach(),
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
        """
        Refine left and right SMPL-X hand poses for a single image.

        target_mp should be shaped [2, 21, 2] or [2, 21, 3].
        Expected order:
            target_mp[0] = left hand
            target_mp[1] = right hand

        Expected joint order:
            Wrist, Index, Middle, Ring, Pinky, Thumb
        """

        global_orient = global_orient.to(self.device).float()
        body_pose = body_pose.to(self.device).float()
        betas = betas.to(self.device).float()
        cam_t = cam_t.to(self.device).float()
        cam_int = cam_int.to(self.device).float()
        bbox_center = bbox_center.to(self.device).float()
        bbox_scale = bbox_scale.to(self.device).float()
        target_mp = target_mp.to(self.device).float()

        # Make hand poses leaf tensors for optimization.
        left_hand_pose = (
            left_hand_pose
            .clone()
            .detach()
            .to(self.device)
            .float()
            .requires_grad_(True)
        )

        right_hand_pose = (
            right_hand_pose
            .clone()
            .detach()
            .to(self.device)
            .float()
            .requires_grad_(True)
        )

        # Keep initial pose as a soft prior.
        left_hand_pose_init = left_hand_pose.clone().detach()
        right_hand_pose_init = right_hand_pose.clone().detach()

        opt = torch.optim.Adam(
            [left_hand_pose, right_hand_pose],
            lr=lr
        )

        for epoch in range(num_epochs):
            opt.zero_grad()

            smplx_output = self.model(
                global_orient=global_orient,
                body_pose=body_pose,
                left_hand_pose=left_hand_pose,
                right_hand_pose=right_hand_pose,
                betas=betas
            )

            body_joints = smplx_output.joints

            # SMPL-X hand joint slices used in your original code.
            lh_joints = smplx_output.joints[:, 25:40, :]
            rh_joints = smplx_output.joints[:, 40:55, :]

            estimate_3d_lh, estimate_3d_rh = self.get_mano_landmarks(
                body_joints,
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

            # Only x/y are used for fitting.
            # Do not treat MediaPipe z as confidence.
            target_2d = target_mp[:, :, :2]

            if save_overlays and epoch == 0:
                _save_overlay(
                    img_path,
                    bbox_center,
                    bbox_scale,
                    estimate_2d.detach().cpu().numpy(),
                    target_2d.detach().cpu().numpy(),
                    "initial_overlay.png"
                )

            total_loss, loss_dict = self.compute_hand_losses(
                estimate_2d_lh=estimate_2d_lh,
                estimate_2d_rh=estimate_2d_rh,
                target_2d=target_2d,
                left_hand_pose=left_hand_pose,
                right_hand_pose=right_hand_pose,
                left_hand_pose_init=left_hand_pose_init,
                right_hand_pose_init=right_hand_pose_init
            )

            if (
                epoch % print_every == 0
                or epoch == num_epochs - 1
            ):
                print(
                    f"Epoch {epoch + 1:04d}/{num_epochs} | "
                    f"total: {loss_dict['total'].item():.4f} | "
                    f"kp2d: {loss_dict['kp2d'].item():.4f} | "
                    f"bone: {loss_dict['bone_dir'].item():.4f} | "
                    f"pose: {loss_dict['pose_prior'].item():.6f} | "
                    f"angle: {loss_dict['angle_limit'].item():.6f}"
                )

            total_loss.backward()
            opt.step()

        if save_overlays:
            with torch.no_grad():
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

                estimate_3d_lh, estimate_3d_rh = self.get_mano_landmarks(
                    body_joints,
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
                    "final_overlay.png"
                )

        return left_hand_pose, right_hand_pose