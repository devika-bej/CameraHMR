"""
diagnose_coords.py

Run this BEFORE any optimization to check if MediaPipe landmarks and MANO
projections live in the same coordinate space.

Usage:
    Call check_coordinate_alignment() with the same inputs you'd pass to refine().
    Read the printed report and look at the saved overlay image.
"""

import torch
import numpy as np
import cv2


MANO_JOINT_NAMES = [
    "Wrist",
    "Index_MCP", "Index_PIP", "Index_DIP", "Index_TIP",
    "Middle_MCP", "Middle_PIP", "Middle_DIP", "Middle_TIP",
    "Ring_MCP", "Ring_PIP", "Ring_DIP", "Ring_TIP",
    "Pinky_MCP", "Pinky_PIP", "Pinky_DIP", "Pinky_TIP",
    "Thumb_MCP", "Thumb_PIP", "Thumb_DIP", "Thumb_TIP",
]


def project_points(points_3d, cam_t, cam_intrinsics):
    """
    Mirrors HandOptimizer.perspective_projection exactly,
    so we're testing the same function used during optimization.
    points_3d: (1, N, 3)
    cam_t:     (1, 3) or (3,)
    cam_intrinsics: (1, 3, 3) or (3, 3)
    Returns: (N, 2) pixel coords
    """
    K = cam_intrinsics
    if K.dim() == 2:
        K = K.unsqueeze(0)
    if cam_t.dim() == 1:
        cam_t = cam_t.unsqueeze(0)

    points_translated = points_3d + cam_t.view(1, 1, 3)
    projected = points_translated / points_translated[:, :, -1].unsqueeze(-1)
    projected = torch.einsum("bij,bkj->bki", K, projected.float())
    return projected[0, :, :2]  # (N, 2)


def check_coordinate_alignment(
    optimizer,          # your HandOptimizer instance
    target_mp,          # (1, 21, 3) raw MediaPipe landmarks
    init_pose,          # (1, 16, 3)
    init_shape,         # (1, 10)
    cam_int,            # (1, 3, 3) or (3, 3)
    cam_t,              # (1, 3) or (3,)
    bbox_center,       # (2,) center of hand bbox in pixel coords
    bbox_scale,        # scalar, relative size of hand bbox
    is_left=True,
    image_path=None,    # optional: path to input image for visual overlay
    image_wh=None,      # (W, H) of image, needed if MediaPipe gave normalized [0,1] coords
    save_overlay=True,
    overlay_path="coord_check_overlay.png"
):
    """
    Step-by-step diagnostic. Checks three common failure modes:
      A. Z-values of MediaPipe landmarks (should be metric-scale, not normalized)
      B. Per-joint pixel error between MANO projection and MediaPipe at initialization
      C. Whether MediaPipe coords need denormalization (are they in [0,1] or pixels?)
    """
    side = 'left' if is_left else 'right'
    model = optimizer.models[side]
    device = optimizer.device

    wrist_pose = init_pose[:, 0, :].clone().detach().to(device)
    hand_pca   = torch.zeros([1, optimizer.num_pca], device=device)
    shape      = init_shape.clone().detach().to(device)

    with torch.no_grad():
        output   = model(hand_pose=hand_pca, global_orient=wrist_pose, betas=shape)
        mano_3d  = optimizer.get_mano_landmarks(output)          # (1, 21, 3)
        mano_proj = project_points(mano_3d, cam_t.to(device), cam_int.to(device))  # (21, 2)

    mp_xy = target_mp[0, :, :2].cpu()   # (21, 2) — whatever units MediaPipe gave you
    mp_z  = target_mp[0, :, 2].cpu()    # (21,)

    mano_proj_cpu = mano_proj.cpu()
    mano_proj_cpu = bbox_center + (mano_proj_cpu - bbox_center) * (bbox_scale * 200.0) # Scale to image space for fair comparison

    # ------------------------------------------------------------------ #
    # CHECK A: MediaPipe Z values
    # ------------------------------------------------------------------ #
    print("\n" + "="*60)
    print("CHECK A — MediaPipe Z-values (expect metric ~0.0–0.1 m range)")
    print("="*60)
    print(f"  Z min:  {mp_z.min().item():.4f}")
    print(f"  Z max:  {mp_z.max().item():.4f}")
    print(f"  Z mean: {mp_z.mean().item():.4f}")
    if mp_z.abs().max() < 0.01:
        print("  ⚠️  Z values near zero — MediaPipe 3D may be in normalized/relative units")
    elif mp_z.abs().max() > 5.0:
        print("  ⚠️  Z values very large — possible unit mismatch (mm vs m?)")
    else:
        print("  ✓  Z values look metric")

    # ------------------------------------------------------------------ #
    # CHECK B: Are MediaPipe XY in pixel space or [0,1] normalized?
    # ------------------------------------------------------------------ #
    print("\n" + "="*60)
    print("CHECK B — MediaPipe XY range (expect pixel coords matching image size)")
    print("="*60)
    print(f"  XY min: {mp_xy.min().item():.2f}  XY max: {mp_xy.max().item():.2f}")
    if mp_xy.max() <= 1.01:
        print("  ⚠️  XY in [0,1] — MediaPipe gave NORMALIZED coords.")
        print("       You must multiply by (image_W, image_H) before comparing to MANO projection.")
        if image_wh is not None:
            W, H = image_wh
            mp_xy = mp_xy * torch.tensor([W, H], dtype=torch.float32)
            print(f"       Auto-scaled to pixel space using image size {W}x{H}")
    else:
        print("  ✓  XY looks like pixel coords")

    # ------------------------------------------------------------------ #
    # CHECK C: Per-joint reprojection error
    # ------------------------------------------------------------------ #
    print("\n" + "="*60)
    print("CHECK C — Per-joint pixel error (MANO init projection vs MediaPipe)")
    print("  < 20px : good init          ")
    print("  20-80px: recoverable        ")
    print("  > 80px : likely frame mismatch or bad initialization")
    print("="*60)

    errors = torch.norm(mano_proj_cpu - mp_xy, dim=-1)  # (21,)
    for i, name in enumerate(MANO_JOINT_NAMES):
        flag = "✓" if errors[i] < 20 else ("⚠️ " if errors[i] < 80 else "❌")
        print(f"  {flag}  {name:<18s}  MANO: ({mano_proj_cpu[i,0]:6.1f}, {mano_proj_cpu[i,1]:6.1f})"
              f"   MP: ({mp_xy[i,0]:6.1f}, {mp_xy[i,1]:6.1f})   err: {errors[i]:.1f}px")

    mean_err = errors.mean().item()
    print(f"\n  Mean error: {mean_err:.1f}px")
    if mean_err > 80:
        print("  ❌ Large systematic offset — likely a coordinate frame or unit problem.")
        _suggest_fixes(mano_proj_cpu, mp_xy)
    elif mean_err > 20:
        print("  ⚠️  Moderate error — initialization is rough but optimization may recover.")
    else:
        print("  ✓  Initialization looks reasonable.")

    # ------------------------------------------------------------------ #
    # VISUAL OVERLAY
    # ------------------------------------------------------------------ #
    if save_overlay and image_path is not None:
        _save_overlay(image_path, mano_proj_cpu, mp_xy, overlay_path)
        print(f"\n  Overlay saved to: {overlay_path}")
    elif save_overlay:
        print("\n  (Provide image_path to get a visual overlay)")

    return {
        "mean_px_error": mean_err,
        "per_joint_error": errors.numpy(),
        "mano_projected": mano_proj_cpu.numpy(),
        "mp_xy_pixels": mp_xy.numpy(),
    }


def _suggest_fixes(mano_proj, mp_xy):
    """Heuristic diagnosis of what kind of mismatch is present."""
    offset = (mp_xy - mano_proj).mean(dim=0)
    print(f"\n  Mean offset (MP - MANO): dx={offset[0]:.1f}px  dy={offset[1]:.1f}px")

    # Y-axis flip check
    mp_y_mean   = mp_xy[:, 1].mean()
    mano_y_mean = mano_proj[:, 1].mean()
    if abs(mp_y_mean + mano_y_mean - 720) < 100 or abs(mp_y_mean - (720 - mano_y_mean)) < 100:
        # rough check for flipped Y around image center
        print("  💡 Possible Y-axis flip — MediaPipe uses top-left origin,")
        print("     MANO projection may use bottom-left. Try: target_mp[:,:,1] = H - target_mp[:,:,1]")

    # Scale check — if MANO is clustered near origin but MP is spread across image
    mano_spread = mano_proj.std()
    mp_spread   = mp_xy.std()
    if mp_spread / (mano_spread + 1e-6) > 5:
        print("  💡 MediaPipe landmarks are much more spread out than MANO projection.")
        print("     Possible unit mismatch: MANO may be projecting in meters, MP in pixels.")
        print("     Check that cam_intrinsics K is in pixel units (fx,fy ~ 500-2000).")


def _save_overlay(image_path, mano_proj, mp_xy, out_path):
    """Draw both sets of landmarks on the image for visual inspection."""
    img = cv2.imread(image_path)
    if img is None:
        print(f"  Could not load image from {image_path}")
        return

    for i in range(len(MANO_JOINT_NAMES)):
        mx, my = int(mano_proj[i, 0]), int(mano_proj[i, 1])
        px, py = int(mp_xy[i, 0]),    int(mp_xy[i, 1])

        # MANO projected = blue circles
        cv2.circle(img, (mx, my), 6, (255, 80, 80), -1)
        # MediaPipe target = green circles
        cv2.circle(img, (px, py), 6, (80, 255, 80), -1)
        # Error line connecting them
        cv2.line(img, (mx, my), (px, py), (0, 0, 255), 1)

    # Legend
    cv2.putText(img, "MANO proj (blue)", (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 80, 80), 2)
    cv2.putText(img, "MediaPipe (green)", (20, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (80, 255, 80), 2)

    cv2.imwrite(out_path, img)


# ------------------------------------------------------------------ #
# Example usage (adapt to your actual data loading)
# ------------------------------------------------------------------ #
if __name__ == "__main__":
    # Assume you have these from your pipeline:
    # optimizer  = HandOptimizer(model_path="path/to/mano")
    # target_mp  = torch.tensor(...)   # (1, 21, 3) from MediaPipe
    # init_pose  = torch.tensor(...)   # (1, 16, 3)
    # init_shape = torch.tensor(...)   # (1, 10)
    # cam_int    = torch.tensor(...)   # (1, 3, 3)
    # cam_t      = torch.tensor(...)   # (1, 3)

    # results = check_coordinate_alignment(
    #     optimizer=optimizer,
    #     target_mp=target_mp,
    #     init_pose=init_pose,
    #     init_shape=init_shape,
    #     cam_int=cam_int,
    #     cam_t=cam_t,
    #     is_left=True,
    #     image_path="left.jpg",
    #     image_wh=(1280, 720),   # your image dimensions
    #     save_overlay=True,
    #     overlay_path="coord_check_overlay.png"
    # )
    pass