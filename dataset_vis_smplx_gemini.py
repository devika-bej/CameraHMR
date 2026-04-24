import os
import cv2
import torch
import smplx
import numpy as np
import argparse
from tqdm import tqdm

# Import components from your existing repository structure
from core.utils.renderer_pyrd import Renderer
from core.constants import SMPL_MODEL_PATH, SMPLX_MODEL_DIR, NUM_BETAS, NUM_BETAS_SMPLX

def visualize_npz(npz_path, image_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"Loading data from {npz_path}...")
    data = np.load(npz_path)
    num_samples = len(data['imgname'])

    # Determine model type based on the keys present in the NPZ output
    is_smplx = 'left_hand_pose' in data.keys()
    
    print(f"Detected model type: {'SMPL-X' if is_smplx else 'SMPL'}")

    # Initialize the body model
    if is_smplx:
        body_model = smplx.SMPLXLayer(
            model_path=SMPLX_MODEL_DIR, 
            num_betas=NUM_BETAS_SMPLX
        ).to(device)
    else:
        body_model = smplx.SMPLLayer(
            model_path=SMPL_MODEL_PATH, 
            num_betas=NUM_BETAS
        ).to(device)

    for i in tqdm(range(num_samples), desc="Rendering images"):
        img_name = data['imgname'][i]
        img_path = os.path.join(image_folder, img_name)

        if not os.path.exists(img_path):
            print(f"Warning: {img_path} not found in {image_folder}. Skipping.")
            continue

        # Load and prep the background image
        img_cv2 = cv2.imread(img_path)
        img_cv2 = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2RGB)
        img_h, img_w, _ = img_cv2.shape

        # Prepare SMPL parameters
        # We reshape them to (1, -1) to ensure compatibility with smplx expectations for batched inputs
        params = {
            # Note: The network output key was 'betas', but it was saved as 'shape' in the NPZ.
            'betas': torch.tensor(data['shape'][i]).view(1, -1).float().to(device)
        }

        if is_smplx:
            params['global_orient'] = torch.tensor(data['global_orient'][i]).view(1, -1).float().to(device)
            params['body_pose'] = torch.tensor(data['body_pose'][i]).view(1, -1).float().to(device)
            params['left_hand_pose'] = torch.tensor(data['left_hand_pose'][i]).view(1, -1).float().to(device)
            params['right_hand_pose'] = torch.tensor(data['right_hand_pose'][i]).view(1, -1).float().to(device)
        else:
            # If standard SMPL, split the (24, 3) pose array into global_orient (1, 3) and body_pose (23, 3)
            pose = data['pose'][i] 
            params['global_orient'] = torch.tensor(pose[0:1]).view(1, -1).float().to(device)
            params['body_pose'] = torch.tensor(pose[1:]).view(1, -1).float().to(device)

        # Reconstruct the mesh
        with torch.no_grad():
            smpl_output = body_model(**params)
            # Extract the first (and only) mesh in the batch
            vertices = smpl_output.vertices[0] 

        # Apply the computed camera translation to the vertices
        cam_t = torch.tensor(data['cam_t'][i]).float().to(device)
        pred_vertices_array = (vertices + cam_t).cpu().numpy()

        # Get focal length from the saved intrinsics matrix
        cam_int = data['cam_int'][i]
        focal_length = cam_int[0, 0]

        # Initialize renderer and generate overlay
        renderer = Renderer(
            focal_length=focal_length, 
            img_w=img_w, 
            img_h=img_h, 
            faces=body_model.faces, 
            same_mesh_color=True
        )
        
        front_view = renderer.render_front_view(pred_vertices_array, bg_img_rgb=img_cv2.copy())

        # Save the rendered image
        fname, img_ext = os.path.splitext(img_name)
        overlay_fname = os.path.join(output_folder, f'{fname}_reconstructed{img_ext}')
        
        out_img = cv2.cvtColor(front_view, cv2.COLOR_RGB2BGR)
        cv2.imwrite(overlay_fname, out_img)

        # Cleanup renderer to free memory
        renderer.delete()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Standalone SMPL/SMPL-X NPZ Visualizer")
    parser.add_argument("--npz_path", type=str, required=True, help="Path to mesh_estimation_output.npz")
    parser.add_argument("--image_folder", type=str, required=True, help="Folder containing the original images")
    parser.add_argument("--output_folder", type=str, required=True, help="Folder to save the rendered outputs")
    
    args = parser.parse_args()
    
    visualize_npz(args.npz_path, args.image_folder, args.output_folder)