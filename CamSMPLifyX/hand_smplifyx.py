import torch
import torch.nn as nn
import smplx

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
        Returns:
            mp_landmarks: (B, 21, 3) - Skeleton in MediaPipe order
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

# --- Optimization Example ---

def optimize_hand(target_mp_keypoints, hand_pose, betas, global_orient, model_path, side):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    mapper = MANOMediapipeMapper(model_path, side=side).to(device)
    
    # Parameters to optimize: Pose (15x3), Shape (10), Global Rot (3), Translation (3)
    hand_pose = torch.tensor(hand_pose, requires_grad=True, device=device)
    betas = torch.tensor(betas, device=device)
    global_orient = torch.tensor(global_orient, device=device)
    transl = torch.zeros((1, 3), requires_grad=True, device=device) # Optional translation
    
    target_mp_keypoints = torch.from_numpy(target_mp_keypoints).float().to(device)
    
    optimizer = torch.optim.Adam([hand_pose, betas, global_orient, transl], lr=0.01)
    
    for i in range(100):
        optimizer.zero_grad()
        
        # Get 21 points from current MANO parameters
        pred_mp = mapper(hand_pose, betas, global_orient, transl)
        
        # Loss: Mean Squared Error between predicted and detected landmarks
        loss = torch.mean((pred_mp - target_mp_keypoints)**2)
        
        # Add Anatomical Regularization (optional but recommended)
        # loss += 0.001 * torch.sum(hand_pose**2) 
        
        loss.backward()
        optimizer.step()
        
        if i % 20 == 0:
            print(f"Iter {i}, Loss: {loss.item():.6f}")

    return hand_pose.detach(), betas.detach()