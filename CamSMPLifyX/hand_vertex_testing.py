import sys
import numpy as np

import os
import pickle
import cv2
import constants
import torch
import trimesh
import numpy as np
from pathlib import Path
from constants import (
    SMPLX_MODEL_DIR,
    NUM_BETAS_SMPLX,
    SMPLX2SMPL,
    DOWNSAMPLE_MAT,
    LOSS_CUT,
    LOW_THRESHOLD,
    HIGH_THRESHOLD,
)
from losses import body_fitting_loss_dense
from utils.smplx_openpose import SMPLX_
from utils.image_utils import crop, read_img, transform
from utils.renderer_cam import render_image_group
from utils.vertex_ids import mano_smplx_lhand_vertex_ids, mano_smplx_rhand_vertex_ids
import mediapipe as mp

IMG_RES = 768


def get_transform(center, scale, res):
    """Generate transformation matrix."""
    h = 200 * scale
    t = torch.zeros(3, 3, device=center.device)  # Ensure device consistency
    t[0, 0] = res[1] / h[0]
    t[1, 1] = res[0] / h[1]
    t[0, 2] = res[1] * (-center[0].float() / h[0] + 0.5)
    t[1, 2] = res[0] * (-center[1].float() / h[1] + 0.5)
    t[2, 2] = 1
    return t


def transform(pts, center, scale, res):
    """Transform pixel locations to a different reference."""
    t = get_transform(center, scale, res)
    ones_column = torch.ones(pts.shape[0], 1, device=pts.device)
    pts = torch.cat(
        (pts, ones_column), dim=1
    )  # Add column of ones for homogeneous coordinates
    new_pts = torch.matmul(t, pts.t()).t()
    new_pts = new_pts[:, :2] / new_pts[:, 2].unsqueeze(
        1
    )  # Normalize homogeneous coordinates
    return new_pts + 1


def j2d_processing(kp, center, scale):
    kp_transformed = transform(kp + 1, center, scale, [IMG_RES, IMG_RES])
    # convert to normalized coordinates
    # kp[:, :-1] = 2.0 * kp[:, :-1] / IMG_RES - 1.0
    return kp_transformed


def perspective_projection(points, translation, cam_intrinsics):
    K = cam_intrinsics
    if translation.ndim == 2:
        translation = translation.squeeze(0)
    points_translated = points + translation.view(1, 3)
    projected_points = points_translated / points_translated[:, -1].unsqueeze(-1)
    projected_points = torch.einsum("ij,kj->ki", K, projected_points.float())
    return projected_points


npz_file = sys.argv[1]
data = np.load(npz_file, allow_pickle=True)
print("Keys in the loaded data:", data.files)
print("Loaded data")

I = 0
img_name = data["imgname"][I]
img_path = os.path.join("./ngt_small_sample", img_name)
global_orient = data["global_orient"][I]
body_pose = np.expand_dims(data["body_pose"][I], axis=0)
left_hand_pose = np.expand_dims(data["left_hand_pose"][I], axis=0)
right_hand_pose = np.expand_dims(data["right_hand_pose"][I], axis=0)
betas = np.expand_dims(data["shape"][I], axis=0)
cam_int = torch.tensor(data["cam_int"][I])
cam_t = torch.tensor(data["cam_t"][I])
center = torch.tensor(data["center"][I])
scale = torch.tensor(data["scale"][I] / 200.0)
dense_kp = data["dense_kp"][I]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
global_orient = torch.tensor(global_orient, device=device)
body_pose = torch.tensor(body_pose, device=device)
left_hand_pose = torch.tensor(left_hand_pose, device=device)
right_hand_pose = torch.tensor(right_hand_pose, device=device)
betas = torch.tensor(betas, device=device)
cam_int = cam_int.to(device)
cam_t = cam_t.to(device)
center = center.to(device)
scale = scale.to(device)
dense_kp = torch.tensor(dense_kp, device=device)
print("Data converted to tensors and moved to device")

smplx_model = SMPLX_(
    model_path=SMPLX_MODEL_DIR,
    num_betas=NUM_BETAS_SMPLX
).to(device)
print("SMPLX model loaded")

smplx_output = smplx_model(
    global_orient=global_orient,
    body_pose=body_pose,
    betas=betas,
    lh_pose=left_hand_pose,
    rh_pose=right_hand_pose,
)
print("SMPLX model forward pass completed")

# Get MANO vertices from SMPLX output
smplx_vertices = smplx_output.vertices.squeeze(0)
lhand_vertices = smplx_vertices[mano_smplx_lhand_vertex_ids.long()]
rhand_vertices = smplx_vertices[mano_smplx_rhand_vertex_ids.long()]

# Project vertices to 2D
lhand_2d = perspective_projection(lhand_vertices, cam_t, cam_int)
rhand_2d = perspective_projection(rhand_vertices, cam_t, cam_int)

# Load image
img = cv2.imread(img_path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=2)
hands_results = hands.process(img)

# Draw MANO vertices
for point in lhand_2d:
    cv2.circle(img, (int(point[0]), int(point[1])), 3, (0, 255, 0), -1)
for point in rhand_2d:
    cv2.circle(img, (int(point[0]), int(point[1])), 3, (0, 0, 255), -1)

# Draw MediaPipe landmarks
if hands_results.multi_hand_landmarks:
    for hand_landmarks, handedness in zip(
        hands_results.multi_hand_landmarks,
        hands_results.multi_handedness,
    ):
        label = handedness.classification[0].label  # "Left" or "Right"
        h, w, _ = img.shape
        if label == 'Right':
            # Red for right hand
            for lm in hand_landmarks.landmark:
                cx, cy = int(lm.x * w), int(lm.y * h)
                cv2.circle(img, (cx, cy), 3, (255, 0, 0), -1)
        else: # Left
            # Yellow for left hand
            for lm in hand_landmarks.landmark:
                cx, cy = int(lm.x * w), int(lm.y * h)
                cv2.circle(img, (cx, cy), 3, (255, 255, 0), -1)

# Save and display the image
output_img_path = "hand_visualization.jpg"
cv2.imwrite(output_img_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
print(f"Visualization saved to {output_img_path}")

# To display the image in a window (optional, might not work in all environments)
# cv2.imshow("Hand Visualization", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
# cv2.waitKey(0)
# cv2.destroyAllWindows()