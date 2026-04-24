import torch
import torch.nn as nn
import smplx
import numpy as np

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

def project_3d_to_2d(landmarks_3d, cam_int):
    """
    Project 3D landmarks to 2D image space using camera intrinsics.
    
    Args:
        landmarks_3d: (B, N, 3) - 3D points in camera space
        cam_int: (3, 3) - Camera intrinsic matrix
    
    Returns:
        landmarks_2d: (B, N, 2) - 2D points in image space
    """
    B, N, _ = landmarks_3d.shape
    
    # Reshape for batch matrix multiplication
    points_3d_flat = landmarks_3d.reshape(B * N, 3, 1)
    
    # Project: p_2d = K @ p_3d
    cam_int_expanded = cam_int.unsqueeze(0).unsqueeze(0).expand(B, N, 3, 3)
    cam_int_flat = cam_int_expanded.reshape(B * N, 3, 3)
    
    proj_2d = torch.bmm(cam_int_flat, points_3d_flat).squeeze(-1)  # (B*N, 3)
    
    # Normalize by Z coordinate (perspective division)
    landmarks_2d = proj_2d[:, :2] / (proj_2d[:, 2:3] + 1e-8)
    landmarks_2d = landmarks_2d.reshape(B, N, 2)
    
    return landmarks_2d


def optimize_hand(target_mp_keypoints, hand_pose, betas, global_orient, cam_int, cam_t, 
                  model_path, side, num_iters=100, lr=0.01):
    """
    Optimize hand pose to match MediaPipe 2D keypoints.
    
    Args:
        target_mp_keypoints: (1, 21, 2 or 3) - Target 2D keypoints (with optional confidence)
        hand_pose: (1, 45) - Initial hand pose
        betas: (1, 10) - Shape parameters (frozen)
        global_orient: (1, 3) - Global orientation (frozen)
        cam_int: (3, 3) - Camera intrinsic matrix
        cam_t: (3,) or (1, 3) - Camera translation from body (initial hand position)
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
    else:
        target_2d = target_mp_keypoints
        confidence = np.ones((target_mp_keypoints.shape[0], target_mp_keypoints.shape[1], 1))
    
    # Convert to torch tensors
    hand_pose_opt = torch.tensor(hand_pose, requires_grad=True, dtype=torch.float32, device=device)
    betas_t = torch.tensor(betas, dtype=torch.float32, device=device)
    global_orient_t = torch.tensor(global_orient, dtype=torch.float32, device=device)
    cam_int_t = torch.tensor(cam_int, dtype=torch.float32, device=device)
    
    # Initialize translation from camera_t (hand position relative to camera)
    if isinstance(cam_t, np.ndarray):
        cam_t = cam_t.reshape(-1)
    cam_t_init = torch.tensor(cam_t, dtype=torch.float32, device=device)
    if cam_t_init.shape[0] == 3:
        cam_t_init = cam_t_init.unsqueeze(0)
    
    transl_opt = torch.tensor(cam_t_init.cpu().numpy(), requires_grad=True, dtype=torch.float32, device=device)
    target_2d_t = torch.tensor(target_2d, dtype=torch.float32, device=device)
    confidence_t = torch.tensor(confidence, dtype=torch.float32, device=device)
    
    # Only optimize hand pose, not shape
    optimizer = torch.optim.Adam([hand_pose_opt, transl_opt], lr=lr)
    
    lambda_kp = 1.0          # 2D keypoint loss weight
    lambda_pose = 0.001      # Hand pose regularization
    lambda_shape = 0.0001    # Shape regularization (soft constraint)
    lambda_transl = 0.0001   # Translation regularization (keep it close to initial)
    
    cam_t_init_detached = cam_t_init.detach()
    
    for i in range(num_iters):
        optimizer.zero_grad()
        
        # Get 3D hand landmarks in camera space
        landmarks_3d = mapper(hand_pose_opt, betas_t, global_orient_t, transl_opt)
        
        # Project 3D landmarks to 2D image space
        landmarks_2d = project_3d_to_2d(landmarks_3d, cam_int_t)
        
        # Keypoint matching loss (only on visible points)
        kp_diff = (landmarks_2d - target_2d_t) * confidence_t
        loss_kp = torch.mean(kp_diff ** 2)
        
        # Hand pose regularization (encourage natural poses)
        loss_pose = torch.mean(hand_pose_opt ** 2)
        
        # Shape regularization (keep shape close to input)
        loss_shape = torch.mean((betas_t - betas_t.detach()) ** 2)
        
        # Translation regularization (keep hand close to initial position)
        loss_transl = torch.mean((transl_opt - cam_t_init_detached) ** 2)
        
        # Total loss
        total_loss = (lambda_kp * loss_kp + 
                     lambda_pose * loss_pose + 
                     lambda_shape * loss_shape + 
                     lambda_transl * loss_transl)
        
        total_loss.backward()
        optimizer.step()
        
        if i % 20 == 0:
            print(f"Iter {i}, Loss: {total_loss.item():.6f} (KP: {loss_kp.item():.6f}, Pose: {loss_pose.item():.6f})")
    
    return hand_pose_opt.detach(), betas_t.detach()