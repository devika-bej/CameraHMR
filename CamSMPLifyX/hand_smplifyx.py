import torch
import smplx
import numpy as np

class HandOptimizer:
    def __init__(self, model_path, device=torch.device("cuda")):
        self.device = device
        # Stage 0: Load MANO with 12 PCA components for biomechanical limits
        self.num_pca = 12
        self.models = {
            'left': smplx.create(model_path, model_type='mano', is_rhand=False, 
                                 use_pca=True, num_pca_comps=self.num_pca, flat_hand_mean=True).to(device),
            'right': smplx.create(model_path, model_type='mano', is_rhand=True, 
                                  use_pca=True, num_pca_comps=self.num_pca, flat_hand_mean=True).to(device)
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

    def perspective_projection(self, points, translation, cam_intrinsics):
        K = cam_intrinsics
        points_translated = points + translation.view(1, 1, 3)
        projected_points = points_translated / points_translated[:, :, -1].unsqueeze(-1)
        projected_points = torch.einsum("bij,bkj->bki", K, projected_points.float())
        return projected_points[:, :, :2]

    def refine(self, target_mp, init_pose, init_shape, cam_int, cam_t, is_left=True):
        side = 'left' if is_left else 'right'
        model = self.models[side]
        target_2d = target_mp[:, :, :2].detach().to(self.device) 
        target_3d = target_mp.detach().to(self.device)
        
        # 1. Variables: Wrist (3D) and Fingers (12D PCA)
        # init_pose is (1, 16, 3); index 0 is the wrist
        wrist_pose = init_pose[:, 0, :].clone().detach().requires_grad_(True)
        hand_pca = torch.zeros([1, self.num_pca], device=self.device, requires_grad=True)
        shape = init_shape.clone().detach().requires_grad_(True)
        
        # Store original wrist for strong regularization
        orig_wrist = init_pose[:, 0, :].clone().detach()

        # --- STAGE 1: Align Wrist Only (100 iterations) ---
        # We lock fingers in a neutral state to prevent the "broken wrist" artifact
        
        initial_estimate = []
        final_estimate = []
        
        opt_s1 = torch.optim.Adam([wrist_pose], lr=0.01)
        for i in range(100):
            opt_s1.zero_grad()
            output = model(hand_pose=hand_pca, global_orient=wrist_pose, betas=shape)
            loss_2d = torch.mean((self.perspective_projection(self.get_mano_landmarks(output), cam_t, cam_int) - target_2d)**2)
            if i == 0:
                initial_estimate = self.get_mano_landmarks(output).detach().cpu().numpy()
            loss_wrist_reg = torch.mean((wrist_pose - orig_wrist)**2) * 50.0 # High weight
            (loss_2d + loss_wrist_reg).backward()
            opt_s1.step()

        # --- STAGE 2: Articulate Fingers (150 iterations) ---
        # Lock the wrist aligned in Stage 1, now solve for the "V" shape/gesture
        opt_s2 = torch.optim.Adam([hand_pca], lr=0.01)
        for _ in range(150):
            opt_s2.zero_grad()
            output = model(hand_pose=hand_pca, global_orient=wrist_pose, betas=shape)
            loss_2d = torch.mean((self.perspective_projection(self.get_mano_landmarks(output), cam_t, cam_int) - target_2d)**2)
            loss_wrist_reg = torch.mean((wrist_pose - orig_wrist)**2) * 50.0 # Keep wrist stable
            loss_pca_prior = torch.mean(hand_pca**2) * 0.5 # Keeps fingers natural
            mano_3d = self.get_mano_landmarks(output)
            root_mano = mano_3d[:, 0:1, :]
            rel_mano_3d = mano_3d - root_mano
            root_target = target_3d[:, 0:1, :]
            rel_target_3d = target_3d - root_target
            scale_mano = torch.norm(rel_mano_3d[:, 9, :], dim=-1, keepdim=True).unsqueeze(-1)
            scale_target = torch.norm(rel_target_3d[:, 9, :], dim=-1, keepdim=True).unsqueeze(-1)
            norm_mano_3d = rel_mano_3d / (scale_mano + 1e-6)
            norm_target_3d = rel_target_3d / (scale_target + 1e-6)
            loss_3d_pose = torch.mean((norm_mano_3d - norm_target_3d)**2)
            weight_3d = 5000.0
            (loss_2d + loss_wrist_reg + loss_pca_prior + weight_3d * loss_3d_pose).backward()
            opt_s2.step()

        # --- STAGE 3: Joint Fine-Tuning (100 iterations) ---
        # Minor adjustments to everything at a lower learning rate
        opt_s3 = torch.optim.Adam([wrist_pose, hand_pca, shape], lr=0.001)
        for i in range(100):
            opt_s3.zero_grad()
            output = model(hand_pose=hand_pca, global_orient=wrist_pose, betas=shape)
            loss_2d = torch.mean((self.perspective_projection(self.get_mano_landmarks(output), cam_t, cam_int) - target_2d)**2)
            if i == 99:
                final_estimate = self.get_mano_landmarks(output).detach().cpu().numpy()
            loss_wrist_reg = torch.mean((wrist_pose - orig_wrist)**2) * 20.0
            loss_pca_prior = torch.mean(hand_pca**2) * 0.2
            mano_3d = self.get_mano_landmarks(output)
            root_mano = mano_3d[:, 0:1, :]
            rel_mano_3d = mano_3d - root_mano
            root_target = target_3d[:, 0:1, :]
            rel_target_3d = target_3d - root_target
            scale_mano = torch.norm(rel_mano_3d[:, 9, :], dim=-1, keepdim=True).unsqueeze(-1)
            scale_target = torch.norm(rel_target_3d[:, 9, :], dim=-1, keepdim=True).unsqueeze(-1)
            norm_mano_3d = rel_mano_3d / (scale_mano + 1e-6)
            norm_target_3d = rel_target_3d / (scale_target + 1e-6)
            loss_3d_pose = torch.mean((norm_mano_3d - norm_target_3d)**2)
            weight_3d = 10.0
            (loss_2d + loss_wrist_reg + loss_pca_prior + weight_3d * loss_3d_pose).backward()
            opt_s3.step()
            
            
        print("Loss between initial and final 2D projections:", np.linalg.norm(initial_estimate - final_estimate))
        print("Loss between initial and target 2D projection:", np.linalg.norm(initial_estimate - target_3d.detach().cpu().numpy()))
        print("Loss between final 2D projection and target:", np.linalg.norm(final_estimate - target_3d.detach().cpu().numpy()))

        # Convert PCA back to raw 15-joint angles for compatibility with optimize.py
        final_output = model(hand_pose=hand_pca, global_orient=wrist_pose, betas=shape)
        # Concatenate optimized wrist + decoded finger joints
        full_pose = torch.cat([wrist_pose, final_output.hand_pose], dim=1)
        
        return full_pose.detach(), shape.detach()