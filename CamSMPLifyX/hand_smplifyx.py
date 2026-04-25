import torch
import smplx
import numpy as np

class HandOptimizer:
    def __init__(self, model_path, device=torch.device("cuda")):
        self.device = device
        # Load MANO models for both hands
        self.models = {
            'left': smplx.create(model_path, model_type='mano', is_rhand=False, use_pca=False, flat_hand_mean=True).to(device),
            'right': smplx.create(model_path, model_type='mano', is_rhand=True, use_pca=False, flat_hand_mean=True).to(device)
        }
        
        # Vertex indices for fingertips (standard MANO mesh)
        self.FINGER_TIPS_V_IDS = [745, 317, 444, 556, 673] # Thumb, Index, Middle, Ring, Pinky
        
        # MediaPipe order mapping to MANO joints
        # Index >= 0 refers to Joint ID; Index < 0 refers to fingertip vertex index
        self.MP_TO_MANO_MAP = [
            0,                      # Wrist
            13, 14, 15, -1,         # Thumb
            1, 2, 3, -2,            # Index
            4, 5, 6, -3,            # Middle
            10, 11, 12, -4,         # Ring
            7, 8, 9, -5              # Pinky
        ]

    def get_mano_landmarks(self, output):
        """Extracts 21 landmarks to match MediaPipe sequence."""
        joints = output.joints  # (1, 16, 3)
        verts = output.vertices # (1, 778, 3)
        landmarks = []
        for idx in self.MP_TO_MANO_MAP:
            if idx >= 0:
                landmarks.append(joints[:, idx, :])
            else:
                v_idx = self.FINGER_TIPS_V_IDS[abs(idx) - 1]
                landmarks.append(verts[:, v_idx, :])
        return torch.stack(landmarks, dim=1) # (1, 21, 3)

    def perspective_projection(self, points, translation, cam_intrinsics):
        """Standard projection as used in cam_smplifyx.py."""
        K = cam_intrinsics
        # points: (1, 21, 3), translation: (1, 3)
        points_translated = points + translation.view(1, 1, 3) # (1, 21, 3)
        projected_points = points_translated / points_translated[:, :, -1].unsqueeze(-1)
        # Apply intrinsics K    
        projected_points = torch.einsum("bij,bkj->bki", K, projected_points.float())
        return projected_points[:, :, :2] # Return (1, 21, 2) pixels

    def refine(self, target_mp, init_pose, init_shape, cam_int, cam_t, is_left=True):
        """Optimizes hand pose to match MediaPipe 2D landmarks."""
        side = 'left' if is_left else 'right'
        model = self.models[side]
        
        # Target: (21, 2) pixels
        target_2d = target_mp[:, :2].unsqueeze(0).to(self.device) 
        
        # Optimize Wrist (Index 0) + Fingers (Index 1-15)
        hand_full_pose = init_pose.clone().detach().requires_grad_(True)
        shape = init_shape.clone().detach().requires_grad_(True)
        
        optimizer = torch.optim.Adam([hand_full_pose, shape], lr=0.01)

        for i in range(150):
            optimizer.zero_grad()
            
            # Forward pass: full_pose (1, 16, 3) -> 1 wrist + 15 fingers
            hand_pose = hand_full_pose[:, 1:]  # Fingers
            hand_pose = hand_pose.reshape(1, 15*3)  # Flatten fingers
            global_orient = hand_full_pose[:, :1]  # Wrist
            global_orient = global_orient.reshape(1, 3)  # Flatten wrist
            output = model(hand_pose=hand_pose, global_orient=global_orient, betas=shape)
            landmarks_3d = self.get_mano_landmarks(output)
            
            # Project to pixel space
            landmarks_2d = self.perspective_projection(landmarks_3d, cam_t, cam_int)
            
            # Loss: Euclidean distance in pixels
            loss_2d = torch.mean((landmarks_2d - target_2d)**2)
            
            # Regularization: Keep pose close to initial estimate
            loss_reg = torch.mean((hand_full_pose - init_pose)**2) * 0.1
            
            total_loss = loss_2d + loss_reg
            total_loss.backward()
            optimizer.step()
            
        return hand_full_pose.detach(), shape.detach()