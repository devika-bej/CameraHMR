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
    points_translated = points + translation.unsqueeze(0)
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

model_joints = smplx_output.joints.detach()
model_verts = smplx_output.vertices.detach()

dense_kp = torch.tensor(dense_kp, device=device, dtype=torch.float32)
dense_kp = (dense_kp + 0.5) * IMG_RES

image_full = cv2.imread(img_path)
image_full = image_full[:, :, ::-1]
img_h, img_w, _ = image_full.shape

vertices3d = smplx_output.vertices
img_h, img_w, _ = image_full.shape

bbox_center = center
bbox_scale = scale

focal_length = cam_int[0, 0].item()

print(image_full.shape, bbox_center, bbox_scale)
render_img = crop(image_full, bbox_center, bbox_scale, [IMG_RES, IMG_RES])
# render_img = image_full.copy()
# h = 200 * bbox_scale[0].item()
# x1 = int(bbox_center[0].item() - h / 2)
# y1 = int(bbox_center[1].item() - h / 2)
# x2 = int(x1 + h)
# y2 = int(y1 + h)
# cv2.rectangle(render_img, (x1, y1), (x2, y2), (255, 255, 255), 2)
print(render_img.shape)

# cv2.imwrite("hand_fitting_result.png", image_full[:, :, ::-1])
cv2.imwrite("hand_fitting_result.png", render_img[:, :, ::-1])