"""
Evaluate estimated SMPL-X hand poses against MediaPipe 2D hand-keypoint
ground truth.

Pipeline
--------
1. Load the estimate file (SMPL-X params + camera + crop info per frame)
   and the MediaPipe file (2D hand keypoints per frame).
2. Align frames between the two files by `imgname` (do NOT assume the two
   files share row order).
3. Run one batched SMPL-X forward pass over all aligned frames to get 3D
   joints.
4. Slice out the 21 hand joints per hand using the SMPL-X<->OpenPose/
   MediaPipe joint index mapping.
5. Project those 3D joints to 2D with the camera intrinsics/translation,
   then map into the same crop-image coordinate space the keypoints were
   annotated in.
6. Compute masked MSE per hand (ignoring missing/low-confidence GT points),
   averaged over all valid frames.

Config you should verify against your own data before trusting numbers:
  - HAND_KEY_FOR_{LEFT,RIGHT}: which MediaPipe array key is the
    *anatomical* left/right hand. This is ambiguous from field names alone
    (anatomical vs image-left/right conventions both exist in the wild) --
    see `sanity_check_hand_assignment` below, which will tell you if the
    mapping looks backwards.
  - CONF_THRESHOLD: minimum confidence for a MediaPipe point to count.
"""

import os
import sys
import numpy as np
import torch

from CamSMPLifyX.utils.smplx_openpose import SMPLX_
from CamSMPLifyX.constants import SMPLX_MODEL_DIR, NUM_BETAS_SMPLX

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
IMG_RES = 768
CONF_THRESHOLD = 0.0

# Which MediaPipe dict key is the anatomical left / right hand.
# FLIP THESE TWO LINES if your MediaPipe export uses image-left/image-right
# instead of anatomical-left/anatomical-right.
HAND_KEY_FOR_LEFT = "mediapipe_kp_left"
HAND_KEY_FOR_RIGHT = "mediapipe_kp_right"

# SMPL-X joint indices -> 21-point OpenPose/MediaPipe hand ordering.
# (Same topology is used for both hands; only the SMPL-X joint set differs.)
LHAND_JOINT_IDX = [20, 37, 38, 39, 66, 25, 26, 27, 67, 28, 29, 30, 68, 34, 35, 36, 69, 31, 32, 33, 70]
RHAND_JOINT_IDX = [21, 52, 53, 54, 71, 40, 41, 42, 72, 43, 44, 45, 73, 49, 50, 51, 74, 46, 47, 48, 75]


# ---------------------------------------------------------------------------
# Step 1: frame alignment
# ---------------------------------------------------------------------------
def align_frames(estimate, mp_data):
    """
    Returns parallel arrays of estimate-indices and mp_data-indices for
    frames present in BOTH files, matched by `imgname`. Order follows the
    estimate file.
    """
    if "imgname" not in estimate or "imgname" not in mp_data:
        raise KeyError(
            "Both the estimate file and the mediapipe file must contain an "
            "'imgname' field to align frames safely. Refusing to fall back "
            "to index alignment, since a silent misalignment there would "
            "invalidate the whole metric."
        )

    def normalize(n):
        # npz can store strings as numpy.bytes_; decode before doing anything else.
        if isinstance(n, bytes):
            n = n.decode("utf-8")
        else:
            n = str(n)
        # estimate file stores full absolute paths, mp_data stores bare
        # filenames -- match on basename so both sides agree.
        return os.path.basename(n)

    est_names = [normalize(n) for n in estimate["imgname"]]
    mp_names = [normalize(n) for n in mp_data["imgname"]]
    mp_lookup = {name: idx for idx, name in enumerate(mp_names)}

    est_idx, mp_idx, unmatched = [], [], 0
    for i, name in enumerate(est_names):
        j = mp_lookup.get(name)
        if j is None:
            unmatched += 1
            continue
        est_idx.append(i)
        mp_idx.append(j)

    if unmatched:
        print(f"WARNING: {unmatched}/{len(est_names)} estimate frames had no "
              f"matching imgname in the mediapipe file; they are excluded.")
    if not est_idx:
        raise ValueError("No frames could be aligned between the two files.")

    return np.array(est_idx), np.array(mp_idx)


# ---------------------------------------------------------------------------
# Step 2: batched SMPL-X forward pass
# ---------------------------------------------------------------------------
def run_smplx_batched(estimate, est_idx, device):
    N = len(est_idx)
    model = SMPLX_(SMPLX_MODEL_DIR, num_betas=NUM_BETAS_SMPLX, use_pca=False).to(device)

    def batched(key):
        return torch.tensor(np.asarray(estimate[key])[est_idx]).float().to(device)

    # The model's own defaults for these (unused) components are sized for
    # its construction-time batch_size (typically 1). If we don't pass them
    # explicitly at batch size N, torch.cat inside the model's forward()
    # fails trying to concat a size-1 default against our size-N tensors.
    num_expr = getattr(model, "num_expression_coeffs", 10)
    zeros = lambda dim: torch.zeros(N, dim, device=device, dtype=torch.float32)

    with torch.no_grad():
        output = model(
            global_orient=batched("global_orient"),
            body_pose=batched("body_pose"),
            left_hand_pose=batched("left_hand_pose"),
            right_hand_pose=batched("right_hand_pose"),
            betas=batched("shape"),
            jaw_pose=zeros(3),
            leye_pose=zeros(3),
            reye_pose=zeros(3),
            expression=zeros(num_expr),
        )
    return output.joints  # (N, J, 3)


# ---------------------------------------------------------------------------
# Step 3: camera projection + crop-space mapping (vectorized over frames)
# ---------------------------------------------------------------------------
def project_to_crop_space(joints_3d, estimate, est_idx, device):
    """
    joints_3d: (N, K, 3) -- already the sliced hand joints, K per hand.
    Returns (N, K, 2) normalized crop-space coordinates in [0, 1]-ish range.
    """
    N, K, _ = joints_3d.shape

    cam_t = torch.tensor(np.asarray(estimate["cam_t"])[est_idx]).float().to(device)      # (N, 3)
    cam_int = torch.tensor(np.asarray(estimate["cam_int"])[est_idx]).float().to(device)  # (N, 3, 3)
    center = torch.tensor(np.asarray(estimate["center"])[est_idx]).float().to(device)    # (N, 2)
    scale = torch.tensor(np.asarray(estimate["scale"])[est_idx]).float().to(device)      # (N,) or (N, 2)

    # Flatten scale if it's strictly (N, 1)
    if scale.dim() == 2 and scale.shape[1] == 1:
        scale = scale.squeeze(1)

    # --- perspective projection ---
    points = joints_3d + cam_t.unsqueeze(1)                 # (N, K, 3)
    z = points[..., 2:3].clamp(min=1e-6)
    points = points / z
    projected = torch.einsum("nij,nkj->nki", cam_int, points)  # (N, K, 3), homogeneous pixel coords
    projected_2d = projected[..., :2]                          # (N, K, 2)

    # --- affine transform into the IMG_RES x IMG_RES crop space ---
    h = 200.0 * scale                                        
    
    # Handle both 1D and 2D scales robustly
    if h.dim() == 1:
        hx = hy = h
    else:
        hx = h[:, 0]
        hy = h[:, 1]

    tx = IMG_RES * (-center[:, 0] / hx + 0.5)                 # (N,)
    ty = IMG_RES * (-center[:, 1] / hy + 0.5)                 # (N,)
    sxy_x = (IMG_RES / hx)                                    # (N,)
    sxy_y = (IMG_RES / hy)                                    # (N,)

    pts = projected_2d + 1.0                                 # match original convention's +1 offset
    crop_x = pts[..., 0] * sxy_x.unsqueeze(1) + tx.unsqueeze(1)
    crop_y = pts[..., 1] * sxy_y.unsqueeze(1) + ty.unsqueeze(1)
    crop_xy = torch.stack([crop_x, crop_y], dim=-1) + 1.0     # (N, K, 2)

    return crop_xy / IMG_RES


# ---------------------------------------------------------------------------
# Step 4: load + normalize MediaPipe ground truth
# ---------------------------------------------------------------------------
def load_mp_hand(mp_data, key, mp_idx, device):
    raw = np.asarray(mp_data[key])[mp_idx].astype(np.float32)
    coords = raw[..., :2] / IMG_RES
    
    # FIX: The 3rd channel is MediaPipe relative Z-depth, not a confidence score.
    # Therefore, a negative value is a valid depth, not a low confidence.
    # We identify missing frames purely by checking if the coordinate is the (0,0) origin padding.
    valid = ~((raw[..., 0] == 0.0) & (raw[..., 1] == 0.0))
    
    return torch.tensor(coords).to(device), torch.tensor(valid).to(device)


# ---------------------------------------------------------------------------
# Step 5: metric
# ---------------------------------------------------------------------------
def masked_mse_per_frame(est_xy, gt_xy, valid_mask):
    """
    est_xy, gt_xy: (N, K, 2). valid_mask: (N, K) bool.
    Returns a (N,) tensor of per-frame MSE (NaN where a frame has zero
    valid keypoints for that hand).
    """
    sq_err = ((est_xy - gt_xy) ** 2).sum(dim=-1)  # (N, K)
    sq_err = torch.where(valid_mask, sq_err, torch.full_like(sq_err, float("nan")))
    with torch.no_grad():
        per_frame = torch.nanmean(sq_err, dim=1)
    return per_frame


# ---------------------------------------------------------------------------
# Optional: catch an anatomical-vs-image left/right mixup automatically
# ---------------------------------------------------------------------------
def sanity_check_hand_assignment(est_l, est_r, gt_l, gt_lv, gt_r, gt_rv):
    """
    Prints a warning if swapping the hand assignment would give a
    substantially lower error -- a strong signal the key mapping is
    backwards for this dataset.
    """
    def mean_err(est, gt, mask):
        e = masked_mse_per_frame(est, gt, mask)
        return torch.nanmean(e).item()

    normal = mean_err(est_l, gt_l, gt_lv) + mean_err(est_r, gt_r, gt_rv)
    swapped = mean_err(est_l, gt_r, gt_rv) + mean_err(est_r, gt_l, gt_lv)

    if swapped < normal * 0.5:
        print("WARNING: swapping left/right hand ground truth gives a much "
              "lower combined MSE than the current HAND_KEY_FOR_LEFT/RIGHT "
              "assignment. Your MediaPipe file likely uses image-left/"
              "image-right rather than anatomical left/right (or vice "
              "versa) -- flip HAND_KEY_FOR_LEFT/RIGHT at the top of this "
              "file.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def evaluate(estimate, mp_data):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    est_idx, mp_idx = align_frames(estimate, mp_data)
    n_frames = len(est_idx)

    joints_3d = run_smplx_batched(estimate, est_idx, device)  # (N, J, 3)
    lhand_3d = joints_3d[:, LHAND_JOINT_IDX, :]
    rhand_3d = joints_3d[:, RHAND_JOINT_IDX, :]

    est_lhand = project_to_crop_space(lhand_3d, estimate, est_idx, device)  # (N, 21, 2)
    est_rhand = project_to_crop_space(rhand_3d, estimate, est_idx, device)  # (N, 21, 2)

    gt_lhand, gt_lhand_valid = load_mp_hand(mp_data, HAND_KEY_FOR_LEFT, mp_idx, device)
    gt_rhand, gt_rhand_valid = load_mp_hand(mp_data, HAND_KEY_FOR_RIGHT, mp_idx, device)

    sanity_check_hand_assignment(
        est_lhand, est_rhand, gt_lhand, gt_lhand_valid, gt_rhand, gt_rhand_valid
    )

    lhand_mse_per_frame = masked_mse_per_frame(est_lhand, gt_lhand, gt_lhand_valid)
    rhand_mse_per_frame = masked_mse_per_frame(est_rhand, gt_rhand, gt_rhand_valid)

    lhand_valid_frames = (~torch.isnan(lhand_mse_per_frame)).sum().item()
    rhand_valid_frames = (~torch.isnan(rhand_mse_per_frame)).sum().item()

    print(f"Total Frames Evaluated:              {n_frames}")
    print(f"Frames with valid L-Hand keypoints:  {lhand_valid_frames}")
    print(f"Frames with valid R-Hand keypoints:  {rhand_valid_frames}")
    print(f"Mean L-Hand MSE: "
          f"{torch.nanmean(lhand_mse_per_frame).item():.6f}" if lhand_valid_frames else
          "Mean L-Hand MSE: N/A")
    print(f"Mean R-Hand MSE: "
          f"{torch.nanmean(rhand_mse_per_frame).item():.6f}" if rhand_valid_frames else
          "Mean R-Hand MSE: N/A")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python mediapipe_metric.py <estimate_file> <mediapipe_file>")
        sys.exit(1)

    estimate = np.load(sys.argv[1], allow_pickle=True)
    mp_data = np.load(sys.argv[2], allow_pickle=True)

    evaluate(estimate, mp_data)
