import numpy as np
import sys
import torch

from CamSMPLifyX.utils.smplx_openpose import SMPLX_
from CamSMPLifyX.constants import (
    SMPLX_MODEL_DIR,
    NUM_BETAS_SMPLX,
)

IMG_RES = 768


def get_transform(center, scale, res):
    h = 200 * scale
    t = torch.zeros(3, 3, device=center.device, dtype=center.dtype)

    t[0, 0] = res[1] / h[0]
    t[1, 1] = res[0] / h[1]
    t[0, 2] = res[1] * (-center[0].float() / h[0] + 0.5)
    t[1, 2] = res[0] * (-center[1].float() / h[1] + 0.5)
    t[2, 2] = 1.0

    return t


def transform(pts, center, scale, res):
    t = get_transform(center, scale, res)

    ones_column = torch.ones(
        pts.shape[0],
        1,
        device=pts.device,
        dtype=pts.dtype,
    )

    pts_homo = torch.cat((pts, ones_column), dim=1)

    new_pts = torch.matmul(t, pts_homo.t()).t()

    new_pts = (
        new_pts[:, :2]
        / new_pts[:, 2].unsqueeze(1)
    )

    return new_pts + 1.0


def j2d_processing(kp, center, scale):
    kp_transformed = transform(
        kp + 1.0,
        center,
        scale,
        [IMG_RES, IMG_RES]
    )
    return kp_transformed


def perspective_projection(points, translation, cam_intrinsics):

    points_translated = (
        points + translation.unsqueeze(0)
    )

    z = (
        points_translated[:, 2]
        .unsqueeze(-1)
        .clamp(min=1e-6)
    )

    projected_points = (
        points_translated / z
    )

    projected_points = torch.einsum(
        "ij,kj->ki",
        cam_intrinsics,
        projected_points.float()
    )

    return projected_points


def compute_mse(est_2d, target_2d):

    valid_mask = (
        (target_2d[:, 0] != 0.0)
        &
        (target_2d[:, 1] != 0.0)
    )

    if valid_mask.sum() == 0:
        return 0.0

    err = (
        est_2d[valid_mask]
        -
        target_2d[valid_mask]
    ) ** 2

    return err.sum(dim=-1).mean().item()


def evaluate_single_view(estimate, mp_data):

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "cpu"
    )

    model = SMPLX_(
        SMPLX_MODEL_DIR,
        num_betas=NUM_BETAS_SMPLX,
        use_pca=False
    ).to(device)

    n_frames = estimate["imgname"].shape[0]

    lhand_errs = []
    rhand_errs = []

    lhand_mapping = [
        20,37,38,39,66,
        25,26,27,67,
        28,29,30,68,
        34,35,36,69,
        31,32,33,70
    ]

    rhand_mapping = [
        21,52,53,54,71,
        40,41,42,72,
        43,44,45,73,
        49,50,51,74,
        46,47,48,75
    ]

    for i in range(n_frames):

        mp_rhand = mp_data['mediapipe_kp_left'][i][:, :2]
        mp_lhand = mp_data['mediapipe_kp_right'][i][:, :2]

        global_orient = torch.tensor(
            estimate["global_orient"][i][None]
        ).float().to(device)

        body_pose = torch.tensor(
            estimate["body_pose"][i][None]
        ).float().to(device)

        left_hand_pose = torch.tensor(
            estimate["left_hand_pose"][i][None]
        ).float().to(device)

        right_hand_pose = torch.tensor(
            estimate["right_hand_pose"][i][None]
        ).float().to(device)

        betas = torch.tensor(
            estimate["shape"][i][None]
        ).float().to(device)

        cam_t = torch.tensor(
            estimate["cam_t"][i]
        ).float().to(device)

        cam_int = torch.tensor(
            estimate["cam_int"][i]
        ).float().to(device)

        center = torch.tensor(
            estimate["center"][i]
        ).float().to(device)

        scale = torch.tensor(
            estimate["scale"][i]
        ).float().to(device)

        output = model(
            global_orient=global_orient,
            body_pose=body_pose,
            left_hand_pose=left_hand_pose,
            right_hand_pose=right_hand_pose,
            betas=betas
        )

        joints_3d = output.joints.squeeze(0)

        lhand_3d = joints_3d[lhand_mapping]
        rhand_3d = joints_3d[rhand_mapping]

        proj_l = perspective_projection(
            lhand_3d,
            cam_t,
            cam_int
        )[:, :2]

        proj_r = perspective_projection(
            rhand_3d,
            cam_t,
            cam_int
        )[:, :2]

        proj_l = (
            j2d_processing(
                proj_l,
                center,
                scale
            ) / IMG_RES
        )

        proj_r = (
            j2d_processing(
                proj_r,
                center,
                scale
            ) / IMG_RES
        )

        mp_l = torch.tensor(
            mp_lhand,
            dtype=torch.float32,
            device=device
        ) / IMG_RES

        mp_r = torch.tensor(
            mp_rhand,
            dtype=torch.float32,
            device=device
        ) / IMG_RES

        mse_l = compute_mse(
            proj_l,
            mp_l
        )

        mse_r = compute_mse(
            proj_r,
            mp_r
        )

        lhand_errs.append(mse_l)
        rhand_errs.append(mse_r)

    mean_l = np.mean(lhand_errs)
    mean_r = np.mean(rhand_errs)

    return (mean_l + mean_r) / 2.0


if __name__ == "__main__":

    if len(sys.argv) != 7:

        print(
            "Usage:\n"
            "python mono_mvae.py "
            "<mono_front.npz> "
            "<mono_left.npz> "
            "<mono_right.npz> "
            "<mp_front.npz> "
            "<mp_left.npz> "
            "<mp_right.npz>"
        )
        sys.exit(1)

    mono_front = np.load(
        sys.argv[1],
        allow_pickle=True
    )

    mono_left = np.load(
        sys.argv[2],
        allow_pickle=True
    )

    mono_right = np.load(
        sys.argv[3],
        allow_pickle=True
    )

    mp_front = np.load(
        sys.argv[4],
        allow_pickle=True
    )

    mp_left = np.load(
        sys.argv[5],
        allow_pickle=True
    )

    mp_right = np.load(
        sys.argv[6],
        allow_pickle=True
    )

    front_err = evaluate_single_view(
        mono_front,
        mp_front
    )

    left_err = evaluate_single_view(
        mono_left,
        mp_left
    )

    right_err = evaluate_single_view(
        mono_right,
        mp_right
    )

    mvae = (
        front_err +
        left_err +
        right_err
    ) / 3.0

    print("\n========== MONOCULAR MVAE ==========")
    print(f"Front Error : {front_err:.6f}")
    print(f"Left Error  : {left_err:.6f}")
    print(f"Right Error : {right_err:.6f}")
    print("------------------------------------")
    print(f"MVAE        : {mvae:.6f}")
    print("====================================")
