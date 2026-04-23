import torch
import torch.nn as nn
import smplx
import numpy as np
from losses import perspective_projection, j2d_processing, gmof

class MANOMediapipeMapper(nn.Module):
    def __init__(self, model_path, side='right'):
        super().__init__()
        # Initialize the MANO model via smplx
        self.mano = smplx.create(
            model_path, 
            model_type='mano', 
            use_pca=False, # Full 15x3 articulation
            flat_hand_mean=True
        )
        
        # Vertex indices for fingertips used in smplx/vertex_ids.py
        #
        if side == 'right':
            self.register_buffer('tip_v_idx', torch.tensor([729, 320, 443, 554, 671], dtype=torch.long)) # Example indices for right hand tips
        else:
            self.register_buffer('tip_v_idx', torch.tensor([744, 320, 443, 554, 671], dtype=torch.long)) # Example indices for left hand tips

    def forward(self, hand_pose, betas, global_orient, transl=None):
        """
        Inputs:
            hand_pose: (B, 45) - 15 joints * 3 axis-angle
            betas: (B, 10) - Shape parameters
            global_orient: (B, 3) - Wrist rotation
            transl: (B, 3) - Translation in camera space
        Returns:
            mp_landmarks: (B, 21, 3) - Skeleton in MediaPipe order (camera space)
        """
        output = self.mano(
            hand_pose=hand_pose, 
            betas=betas, 
            global_orient=global_orient,
            transl=transl,
            return_verts=True
        )
        
        verts = output.vertices # (B, 778, 3)
        joints = output.joints   # (B, 16, 3)
        
        # Extract fingertip vertices
        tips = verts[:, self.tip_v_idx, :]
        
        # Reorder and interleave joints/tips to match MediaPipe 21-landmark convention
        # B = batch size
        mp_landmarks = torch.zeros((joints.shape[0], 21, 3), device=joints.device)
        
        mp_landmarks[:, 0, :] = joints[:, 0, :]      # Wrist
        mp_landmarks[:, 1:4, :] = joints[:, 13:16, :] # Thumb joints
        mp_landmarks[:, 4, :] = tips[:, 0, :]         # Thumb Tip
        mp_landmarks[:, 5:8, :] = joints[:, 1:4, :]   # Index joints
        mp_landmarks[:, 8, :] = tips[:, 1, :]         # Index Tip
        mp_landmarks[:, 9:12, :] = joints[:, 4:7, :]  # Middle joints
        mp_landmarks[:, 12, :] = tips[:, 2, :]        # Middle Tip
        mp_landmarks[:, 13:16, :] = joints[:, 10:13, :] # Ring joints
        mp_landmarks[:, 16, :] = tips[:, 3, :]        # Ring Tip
        mp_landmarks[:, 17:20, :] = joints[:, 7:10, :] # Pinky joints
        mp_landmarks[:, 20, :] = tips[:, 4, :]        # Pinky Tip
        
        return mp_landmarks


def optimize_hand(target_mp_keypoints, hand_pose, betas, global_orient, cam_int, cam_t, 
                  center, scale,
                  model_path, side, num_iters=200, lr=0.02):
    """
    Optimize hand pose to match MediaPipe 2D keypoints using proper camera projection and robust losses.
    
    Args:
        target_mp_keypoints: (1, 21, 2 or 3) - Target 2D keypoints (with optional confidence)
        hand_pose: (1, 45) - Initial hand pose
        betas: (1, 10) - Shape parameters (frozen)
        global_orient: (1, 3) - Global orientation (frozen)
        cam_int: (3, 3) - Camera intrinsic matrix
        cam_t: (3,) or (1, 3) - Camera translation in camera frame (FIXED - from body)
        center: torch.Tensor - Image center for j2d_processing
        scale: torch.Tensor - Image scale for j2d_processing
        model_path: str - Path to MANO model
        side: str - 'left' or 'right'
        num_iters: int - Number of optimization iterations
        lr: float - Learning rate
    
    Returns:
        hand_pose_opt: (1, 45) - Optimized hand pose
        betas: (1, 10) - Shape parameters (unchanged)
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    mapper = MANOMediapipeMapper(model_path, side=side).to(device)
    
    # Extract 2D keypoints and confidence if available
    if target_mp_keypoints.shape[-1] == 3:
        target_2d = target_mp_keypoints[:, :, :2]
        confidence = target_mp_keypoints[:, :, 2:]
        # Filter out points with low confidence
        confidence = (confidence > 0.5).astype(np.float32)
    else:
        target_2d = target_mp_keypoints
        confidence = np.ones((target_mp_keypoints.shape[0], target_mp_keypoints.shape[1], 1))
    
    num_visible_kps = confidence.sum()
    print(f"[{side.upper()}] Target 2D shape: {target_2d.shape}, Visible keypoints: {num_visible_kps}/{confidence.size}")
    print(f"[{side.upper()}] Target 2D range: min={target_2d.min():.2f}, max={target_2d.max():.2f}")
    
    # Convert to torch tensors
    hand_pose_opt = torch.tensor(hand_pose, requires_grad=True, dtype=torch.float32, device=device)
    betas_t = torch.tensor(betas, dtype=torch.float32, device=device)
    global_orient_t = torch.tensor(global_orient, dtype=torch.float32, device=device)
    
    # Ensure camera parameters are on correct device
    if isinstance(cam_int, np.ndarray):
        cam_int = torch.from_numpy(cam_int).float()
    cam_int_t = cam_int.to(device)
    
    # FIX 1: Camera translation is FIXED from body, NOT optimized
    if isinstance(cam_t, np.ndarray):
        cam_t = cam_t.reshape(-1)
    cam_t_fixed = torch.tensor(cam_t, dtype=torch.float32, device=device)
    if cam_t_fixed.shape[0] == 3:
        cam_t_fixed = cam_t_fixed.unsqueeze(0)
    
    # Target data
    target_2d_t = torch.tensor(target_2d, dtype=torch.float32, device=device)
    confidence_t = torch.tensor(confidence, dtype=torch.float32, device=device)
    
    # Ensure center and scale are on device
    if isinstance(center, torch.Tensor):
        center = center.to(device)
    else:
        center = torch.tensor(center, dtype=torch.float32, device=device)
    
    if isinstance(scale, torch.Tensor):
        scale = scale.to(device)
    else:
        scale = torch.tensor(scale, dtype=torch.float32, device=device)
    
    print(f"[{side.upper()}] Camera intrinsics shape: {cam_int_t.shape}")
    print(f"[{side.upper()}] Fixed camera translation: {cam_t_fixed.cpu().numpy()}")
    
    # FIX 2: Only optimize hand pose, NOT translation
    optimizer = torch.optim.Adam([hand_pose_opt], lr=lr)
    
    # FIX 3: Better loss weight balance
    kp_weight = 5.0               # 2D keypoint reprojection (INCREASED - prioritize keypoint matching)
    pose_prior_weight = 0.001     # Hand pose prior (DECREASED - allow fingers to move)
    
    sigma = 100                   # GMOF sigma for robust estimation
    
    print(f"[{side.upper()}] Starting optimization with {num_iters} iterations...")
    print(f"[{side.upper()}] Loss weights: KP={kp_weight}, Pose={pose_prior_weight}")
    
    for iteration in range(num_iters):
        optimizer.zero_grad()
        
        # Get 3D hand landmarks in camera space (B, 21, 3)
        landmarks_3d = mapper(hand_pose_opt, betas_t, global_orient_t, cam_t_fixed)
        
        # Project 3D landmarks to 2D using camera intrinsics
        joints_2d_full_image = perspective_projection(landmarks_3d[0], cam_t_fixed.squeeze(0), cam_int_t)
        # Apply j2d processing (normalization and transformation)
        projected_keypoints = j2d_processing(joints_2d_full_image, center, scale)
        
        # Keypoint matching loss using robust GMOF
        kp_error = gmof(projected_keypoints - target_2d_t[0], sigma)
        reprojection_loss = kp_weight * (confidence_t[0] * kp_error.sum(dim=-1)).sum()
        
        # FIX 4: Weak regularization on pose to allow natural deformation
        pose_loss = pose_prior_weight * torch.mean(hand_pose_opt ** 2)
        
        # Total loss (NO translation loss)
        total_loss = reprojection_loss + pose_loss
        
        total_loss.backward()
        optimizer.step()
        
        if iteration % 20 == 0 or iteration == num_iters - 1:
            print(f"[{side.upper()}] Iter {iteration:3d}, Loss: {total_loss.item():.6f} | KP: {reprojection_loss.item():.6f} | Pose: {pose_loss.item():.6f}")
        
        if iteration == num_iters - 1:
            # Compute final projected keypoints for debugging
            with torch.no_grad():
                landmarks_3d_final = mapper(hand_pose_opt, betas_t, global_orient_t, cam_t_fixed)
                joints_2d_final = perspective_projection(landmarks_3d_final[0], cam_t_fixed.squeeze(0), cam_int_t)
                proj_final = j2d_processing(joints_2d_final, center, scale)
                final_error = torch.mean(torch.abs(proj_final - target_2d_t[0]) * confidence_t[0])
                print(f"[{side.upper()}] Final mean projected error: {final_error.item():.2f} pixels")
    
    print(f"[{side.upper()}] Optimization complete!")
    
    return hand_pose_opt.detach(), betas_t.detach()