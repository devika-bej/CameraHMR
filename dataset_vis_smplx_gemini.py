import os
import argparse
import cv2
import torch
import numpy as np
from smplx import SMPLX
from core.utils.renderer_pyrd import Renderer


def make_parser():
    parser = argparse.ArgumentParser(description='CameraHMR dataset visualization (SMPLX)')
    parser.add_argument("--image_folder", type=str, default='data/training-images',
        help="Path to input image folder.")
    parser.add_argument("--output_folder", type=str, default='.',
        help="Path to folder output folder.")
    parser.add_argument("--npz_path", type=str, default='data/training-labels/aic-release.npz',
        help="Path to folder output folder.")
    parser.add_argument("--ind", type=int, default=0,
        help="index of npz file")
    return parser


def load_smplx_model(model_folder, gender="neutral", num_betas=16):
    return SMPLX(
        model_folder,
        model_type='smplx',
        gender=gender,
        ext='npz',
        num_betas=num_betas,
        use_pca=False, # We are passing explicit joint rotations, not PCA coeffs
    )


def load_data(npz_path, image_folder, ind):
    data = np.load(npz_path, allow_pickle=True)
    
    img_name_raw = data['imgname'][ind]
    # Handle byte-string decodings if necessary for numpy string arrays
    if isinstance(img_name_raw, bytes):
        img_name_raw = img_name_raw.decode('utf-8')
        
    # img_path = os.path.join(image_folder, img_name_raw.replace('aic-train', 'aic-train-vitpose'))
    img_path = img_name_raw
    
    return {
        "img_path": img_path,
        "translations": data['cam_t'][ind],           # Changed from trans_cam
        "camera_intrinsics": data['cam_int'][ind], 
        "global_orient": data['global_orient'][ind],  # Separated out from pose
        "body_pose": data['body_pose'][ind],          # Separated out from pose
        "left_hand_pose": data['left_hand_pose'][ind],
        "right_hand_pose": data['right_hand_pose'][ind],
        "shape": data['shape'][ind],
    }


def render_model(renderer, model_output, img, outdir, file_name_suffix=""):
    front_view = renderer.render_front_view(model_output.vertices, bg_img_rgb=img)
    side_view = renderer.render_side_view(model_output.vertices)
    final_img = np.hstack([img, front_view, side_view])

    overlay_file_name = os.path.join(outdir, f"{file_name_suffix}.png")
    cv2.imwrite(overlay_file_name, final_img)
    print(f"Overlay saved at: {overlay_file_name}")


def main():

    parser = make_parser()
    args = parser.parse_args()
    
    # Paths and constants
    MODEL_FOLDER = 'data/models/smplx_neutral_head/models_lockedhead/smplx/' # Ensure this points to SMPLX models
    IMAGE_FOLDER = args.image_folder
    NPZ_PATH = args.npz_path
    OUTPUT_DIR = args.output_folder
    ind = args.ind

    # Load SMPLX model. The npz shape parameter is (3, 16), so num_betas=16
    smplx_neutral = load_smplx_model(MODEL_FOLDER, num_betas=16)

    # Load data from npz
    data = load_data(NPZ_PATH, IMAGE_FOLDER, ind)

    # Load image
    img = cv2.imread(data["img_path"])
    if img is None:
        raise FileNotFoundError(f"Image not found: {data['img_path']}")
    print(f"Image loaded: {data['img_path']}")

    img_h, img_w, _ = img.shape

    # Extract parameters
    translations = data["translations"]
    camera_intrinsics = data["camera_intrinsics"]
    global_orient = data["global_orient"]
    body_pose = data["body_pose"]
    left_hand_pose = data["left_hand_pose"]
    right_hand_pose = data["right_hand_pose"]
    shape = data["shape"]

    with torch.no_grad():
        # Run SMPLX model
        model_output = smplx_neutral(
            betas=torch.tensor(shape).unsqueeze(0).float(),
            global_orient=torch.tensor(global_orient).view(1, 3).float(),
            body_pose=torch.tensor(body_pose).view(1, -1).float(),            # Flatten (21, 3) to 63
            left_hand_pose=torch.tensor(left_hand_pose).view(1, -1).float(),  # Flatten (15, 3) to 45
            right_hand_pose=torch.tensor(right_hand_pose).view(1, -1).float(),# Flatten (15, 3) to 45
            transl=torch.tensor(translations).unsqueeze(0).float(),
        )

    # Initialize renderer
    focal_length = camera_intrinsics[0, 0]
    renderer = Renderer(
        focal_length=focal_length,
        img_w=img_w,
        img_h=img_h,
        faces=smplx_neutral.faces,
        same_mesh_color=True,
    )

    # Render and save overlay
    render_model(renderer, model_output, img, OUTPUT_DIR, file_name_suffix="overlay_smplx")

if __name__ == "__main__":
    main()