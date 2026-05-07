import os
import cv2
import torch
import smplx  # Official library
import sys
import numpy as np
from tqdm import tqdm
from scipy.spatial.transform import Rotation as R

# Import renderer from your project structure
from core.utils.renderer_pyrd import Renderer
from core.constants import SMPLX_MODEL_DIR, SMPL_MODEL_PATH # Import your paths

def aa_to_rotmat(axis_angle):
    """
    Helper function to convert axis-angle vectors to 3x3 rotation matrices.
    Input: numpy array of shape (..., 3)CamSMPLifyX/
    Output: numpy array of shape (..., 3, 3)
    """
    original_shape = axis_angle.shape
    # Flatten to a list of 3D vectors
    flat_aa = axis_angle.reshape(-1, 3)
    # Convert to rotation matrices
    rot_mats = R.from_rotvec(flat_aa).as_matrix()
    # Reshape back to original dimensions + (3, 3)
    return rot_mats.reshape(original_shape[:-1] + (3, 3))

def visualize_npz_standard(image_folder, npz_path, output_folder, model_type='smplx'):
    os.makedirs(output_folder, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 1. Load the NPZ data
    print(f"Loading data from {npz_path}...")
    data = np.load(npz_path)
    imgnames = data['imgname']
    num_images = len(imgnames)
    num_betas = data['shape'].shape[1]

    # 2. Initialize the Official SMPL-X/SMPL Layer
    print("Initializing official SMPL/SMPL-X Layer...")
    if model_type == 'smplx':
        body_model = smplx.SMPLXLayer(
            model_path=SMPLX_MODEL_DIR, 
            num_betas=num_betas,
            use_pca=False # Tell it we are providing full hand poses, not PCA
        ).to(device)
    else:
        body_model = smplx.SMPLLayer(
            model_path=SMPL_MODEL_PATH, 
            num_betas=num_betas
        ).to(device)
        
    body_model.eval()

    # 3. Process each image and render
    print(f"Starting visualization for {num_images} images...")
    for i in tqdm(range(num_images)):
        img_name = os.path.basename(imgnames[i])
        img_path = os.path.join(image_folder, img_name)
        
        img_cv2 = cv2.imread(img_path)
        if img_cv2 is None:
            print(f"Warning: Could not read {img_path}. Skipping.")
            continue
            
        img_cv2 = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2RGB)
        img_h, img_w, _ = img_cv2.shape

        # Retrieve Camera Info
        cam_t = torch.tensor(data['cam_t'][i:i+1]).to(device)
        cam_int = data['cam_int'][i]
        focal_length = float(cam_int[0, 0])

        kwargs = {}
        if model_type == 'smplx':
            # Extract axis-angles, convert to rot matrices via scipy, and convert to torch tensors
            # Resulting shapes will be (1, J, 3, 3) which is exactly what SMPLXLayer wants
            kwargs['global_orient'] = torch.tensor(aa_to_rotmat(data['global_orient'][i:i+1])).float().to(device)
            kwargs['body_pose'] = torch.tensor(aa_to_rotmat(data['body_pose'][i:i+1])).float().to(device)
            kwargs['left_hand_pose'] = torch.tensor(aa_to_rotmat(data['left_hand_pose'][i:i+1])).float().to(device)
            kwargs['right_hand_pose'] = torch.tensor(aa_to_rotmat(data['right_hand_pose'][i:i+1])).float().to(device)
            kwargs['betas'] = torch.tensor(data['shape'][i:i+1]).float().to(device)
        else:
            pose = data['pose'][i:i+1] # Contains global_orient + body_pose
            kwargs['global_orient'] = torch.tensor(aa_to_rotmat(pose[:, :1, :])).float().to(device)
            kwargs['body_pose'] = torch.tensor(aa_to_rotmat(pose[:, 1:, :])).float().to(device)
            kwargs['betas'] = torch.tensor(data['shape'][i:i+1]).float().to(device)

        # 4. Generate Mesh
        with torch.no_grad():
            output = body_model(**kwargs)
            vertices = output.vertices[0]

        # Apply camera translation
        pred_vertices_array = (vertices + cam_t[0]).cpu().numpy()

        # 5. Render
        renderer = Renderer(
            focal_length=focal_length, 
            img_w=img_w, 
            img_h=img_h, 
            faces=body_model.faces, 
            same_mesh_color=True
        )
        
        front_view = renderer.render_front_view(
            np.expand_dims(pred_vertices_array, 0), 
            bg_img_rgb=img_cv2.copy()
        )
        renderer.delete()

        # 6. Save Image
        fname, img_ext = os.path.splitext(img_name)
        overlay_fname = os.path.join(output_folder, f'{fname}_npz_overlay{img_ext}')
        front_view_safe = np.clip(front_view, 0, 255).astype(np.uint8)
        cv2.imwrite(overlay_fname, cv2.cvtColor(front_view_safe, cv2.COLOR_RGB2BGR))
        print("imwriting to ", overlay_fname)

    print(f"Done! Visualizations saved to {output_folder}")

if __name__ == "__main__":
    # Example usage:
    # Set these to your respective paths
    IMAGE_DIR = sys.argv[1]
    NPZ_FILE = sys.argv[2]
    OUTPUT_DIR = sys.argv[3]
    
    visualize_npz_standard(
        image_folder=IMAGE_DIR, 
        npz_path=NPZ_FILE, 
        output_folder=OUTPUT_DIR, 
        model_type='smplx'
    )