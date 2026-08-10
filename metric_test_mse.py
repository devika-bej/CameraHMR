import os
import sys
import numpy as np
import torch

from CamSMPLifyX.utils.smplx_openpose import SMPLX_
from CamSMPLifyX.constants import SMPLX_MODEL_DIR, NUM_BETAS_SMPLX

IMG_RES = 768


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

    ones_column = torch.ones(pts.shape[0], 1, device=pts.device, dtype=pts.dtype)

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
    projected_points = torch.einsum("ij,kj->ki", K, projected_points.float())

    return projected_points


def align_frames(estimate, mediapipe):
    est_idx = []
    mp_idx = []

    est_names = [os.path.basename(f) for f in estimate["imgname"]]
    mp_names = [os.path.basename(f) for f in mediapipe["imgname"]]

    for i, est_name in enumerate(est_names):
        if est_name in mp_names:
            est_idx.append(i)
            mp_idx.append(mp_names.index(est_name))

    return est_idx, mp_idx


def get_mano_landmarks(left_wrist, right_wrist, lh_joints, rh_joints, vertices):
    landmarks_lh = []
    landmarks_rh = []

    MP_TO_MANO_MAP = [
        100,
        12,
        13,
        14,
        -1,
        0,
        1,
        2,
        -2,
        3,
        4,
        5,
        -3,
        9,
        10,
        11,
        -4,
        6,
        7,
        8,
        -5,
    ]

    FINGER_TIPS_V_IDS_LH = [5361, 4933, 5058, 5169, 5286]
    FINGER_TIPS_V_IDS_RH = [8079, 7669, 7794, 7905, 8022]

    for idx in MP_TO_MANO_MAP:
        if idx == 100:
            landmarks_lh.append(left_wrist)
            landmarks_rh.append(right_wrist)

        elif idx >= 0:
            landmarks_lh.append(lh_joints[:, idx, :])
            landmarks_rh.append(rh_joints[:, idx, :])

        else:
            v_idx = abs(idx) - 1

            landmarks_lh.append(vertices[:, FINGER_TIPS_V_IDS_LH[v_idx], :])

            landmarks_rh.append(vertices[:, FINGER_TIPS_V_IDS_RH[v_idx], :])

    landmarks_lh = torch.stack(landmarks_lh, dim=1)
    landmarks_rh = torch.stack(landmarks_rh, dim=1)

    return landmarks_lh, landmarks_rh


def get_estimate_2d(estimate, est_idx, device):
    n = len(est_idx)

    model = SMPLX_(
        model_path=SMPLX_MODEL_DIR, num_betas=NUM_BETAS_SMPLX, use_pca=False
    ).to(device)

    with torch.no_grad():
        output = [
            model(
                global_orient=torch.tensor(
                    np.expand_dims(estimate["global_orient"][i], axis=0)
                )
                .float()
                .to(device),
                body_pose=torch.tensor(np.expand_dims(estimate["body_pose"][i], axis=0))
                .float()
                .to(device),
                betas=torch.tensor(np.expand_dims(estimate["shape"][i], axis=0))
                .float()
                .to(device),
                left_hand_pose=torch.tensor(
                    np.expand_dims(estimate["left_hand_pose"][i], axis=0)
                )
                .float()
                .to(device),
                right_hand_pose=torch.tensor(
                    np.expand_dims(estimate["right_hand_pose"][i], axis=0)
                )
                .float()
                .to(device),
            )
            for i in est_idx
        ]

        lh_3d, rh_3d = [], []
        for i, out in enumerate(output):
            vertices = out.vertices
            joints = out.joints

            left_wrist = joints[:, 19, :]
            right_wrist = joints[:, 20, :]

            lh_joints = joints[:, 21:46, :]
            rh_joints = joints[:, 46:71, :]

            lh_landmarks, rh_landmarks = get_mano_landmarks(
                left_wrist, right_wrist, lh_joints, rh_joints, vertices
            )

            lh_3d.append(lh_landmarks)
            rh_3d.append(rh_landmarks)

        lh_2d = torch.stack(
            [
                j2d_processing(
                    perspective_projection(
                        lh_3d[i][0],
                        torch.tensor(estimate["cam_t"][i]).to(device).float(),
                        torch.tensor(estimate["cam_int"][i]).unsqueeze(0).to(device).float()[0],
                    )[:, :2],
                    torch.tensor(estimate["center"][i]).to(device).float(),
                    torch.tensor(estimate["scale"][i]).to(device).float(),
                )
                for i in est_idx
            ]
        ) / IMG_RES
        rh_2d = torch.stack(
            [
                j2d_processing(
                    perspective_projection(
                        rh_3d[i][0],
                        torch.tensor(estimate["cam_t"][i]).to(device).float(),
                        torch.tensor(estimate["cam_int"][i]).unsqueeze(0).to(device).float()[0],
                    )[:, :2],
                    torch.tensor(estimate["center"][i]).to(device).float(),
                    torch.tensor(estimate["scale"][i]).to(device).float(),
                )
                for i in est_idx
            ]
        ) / IMG_RES
    return lh_2d, rh_2d


def get_mediapipe_2d(mediapipe, mp_idx, device):
    lh_mp = torch.tensor(mediapipe["mediapipe_kp_left"][mp_idx]).to(device).float()[..., :2] / IMG_RES
    rh_mp = torch.tensor(mediapipe["mediapipe_kp_right"][mp_idx]).to(device).float()[..., :2] / IMG_RES
    valid_lh = ~((lh_mp[..., 0] == 0) & (lh_mp[..., 1] == 0))
    valid_rh = ~((rh_mp[..., 0] == 0) & (rh_mp[..., 1] == 0))
    return lh_mp, rh_mp, valid_lh, valid_rh


def mse_loss(pred, target, valid_mask):
    sq_err = ((pred - target) ** 2).sum(dim=-1)
    sq_err = torch.where(valid_mask, sq_err, torch.full_like(sq_err, float("nan")))
    with torch.no_grad():
        per_frame_mse = torch.nanmean(sq_err, dim=-1)
    return per_frame_mse


def evaluate(estimate, mediapipe):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    est_idx, mp_idx = align_frames(estimate, mediapipe)
    n_frames = len(est_idx)

    lh_2d, rh_2d = get_estimate_2d(estimate, est_idx, device)
    lh_mp, rh_mp, valid_lh, valid_rh = get_mediapipe_2d(mediapipe, mp_idx, device)

    lh_mse = mse_loss(lh_2d, lh_mp, valid_lh)
    rh_mse = mse_loss(rh_2d, rh_mp, valid_rh)

    lh_valid = (~torch.isnan(lh_mse)).sum().item()
    rh_valid = (~torch.isnan(rh_mse)).sum().item()

    print("Total frames:", n_frames)
    print("Valid Left Hand frames:", lh_valid)
    print("Valid Right Hand frames:", rh_valid)
    print("Left Hand MSE:", lh_mse.mean().item())
    print("Right Hand MSE:", rh_mse.mean().item())


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python metric_test_mse.py <estimate_file> <mediapipe_file>")
        sys.exit(1)

    estimate = np.load(sys.argv[1], allow_pickle=True)
    mediapipe = np.load(sys.argv[2], allow_pickle=True)

    evaluate(estimate, mediapipe)
