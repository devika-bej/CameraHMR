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

    t = torch.zeros(
        3,
        3,
        device=center.device,
        dtype=center.dtype
    )

    t[0, 0] = res[1] / h[0]
    t[1, 1] = res[0] / h[1]

    t[0, 2] = res[1] * (
        -center[0].float() / h[0] + 0.5
    )

    t[1, 2] = res[0] * (
        -center[1].float() / h[1] + 0.5
    )

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

    pts_homo = torch.cat(
        (pts, ones_column),
        dim=1
    )

    new_pts = torch.matmul(
        t,
        pts_homo.t()
    ).t()

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


def perspective_projection(
    points,
    translation,
    cam_intrinsics
):

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
        return None

    err = (
        est_2d[valid_mask]
        -
        target_2d[valid_mask]
    ) ** 2

    return err.sum(dim=-1).mean().item()


def evaluate(
    estimate,
    mp_front,
    mp_left,
    mp_right
):

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

    frame_mvae = []

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

    all_views = [
        mp_front,
        mp_left,
        mp_right
    ]

    for i in range(n_frames):

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

        weighted_error = 0.0
        weight_sum = 0.0

        for mp_data in all_views:

            mp_r = torch.tensor(
                mp_data["mediapipe_kp_left"][i][:, :2],
                dtype=torch.float32,
                device=device
            ) / IMG_RES

            mp_l = torch.tensor(
                mp_data["mediapipe_kp_right"][i][:, :2],
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

            if mse_l is None:
                mse_l = 0.0

            if mse_r is None:
                mse_r = 0.0

            valid_l = (
                (mp_l[:,0] != 0)
                &
                (mp_l[:,1] != 0)
            ).sum().item()

            valid_r = (
                (mp_r[:,0] != 0)
                &
                (mp_r[:,1] != 0)
            ).sum().item()

            weight = valid_l + valid_r

            view_error = (
                mse_l + mse_r
            ) / 2.0

            weighted_error += (
                weight * view_error
            )

            weight_sum += weight

        if weight_sum > 0:
            frame_mvae.append(
                weighted_error / weight_sum
            )

    print(
        f"Frames evaluated: {len(frame_mvae)}"
    )

    print(
        f"MVAE: {np.mean(frame_mvae):.6f}"
    )


if __name__ == "__main__":

    if len(sys.argv) != 5:

        print(
            "Usage:\n"
            "python mvae_metric.py "
            "<estimate.npz> "
            "<mp_front.npz> "
            "<mp_left.npz> "
            "<mp_right.npz>"
        )

        sys.exit(1)

    estimate = np.load(
        sys.argv[1],
        allow_pickle=True
    )

    mp_front = np.load(
        sys.argv[2],
        allow_pickle=True
    )

    mp_left = np.load(
        sys.argv[3],
        allow_pickle=True
    )

    mp_right = np.load(
        sys.argv[4],
        allow_pickle=True
    )

    evaluate(
        estimate,
        mp_front,
        mp_left,
        mp_right
    )
