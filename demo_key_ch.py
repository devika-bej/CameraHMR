import sys
import cv2
import os
import numpy as np

# 1. Add OpenPose library to system path
openpose_build_path = '../src/openpose/build/python'

try:
    sys.path.append(openpose_build_path)
    from openpose import pyopenpose as op
except ImportError as e:
    print(f"Error: OpenPose library could not be found. \n{e}")
    sys.exit(-1)

def initialize_openpose(openpose_dir):
    """Initializes OpenPose wrapper once."""
    params = dict()
    params["model_folder"] = os.path.join(openpose_dir, "models/")
    params["hand"] = True
    params["display"] = 0
    params["render_pose"] = 0

    opWrapper = op.WrapperPython()
    opWrapper.configure(params)
    opWrapper.start()
    return opWrapper

def extract_valid_pose(kp_data, w, h):
    """Helper to safely extract and normalize keypoints, returning None if missing."""
    if kp_data is not None and len(kp_data.shape) > 0 and kp_data.shape[0] > 0:
        # Grabbing [:, :3] to include X, Y, and Confidence.
        # Dividing by [w, h, 1] keeps confidence unchanged while normalizing X, Y.
        pose = np.array(kp_data[0][:, :3]) / [w, h, 1.0]
        return pose * [768.0, 768.0, 1.0]
    return None

def fill_missing_poses(pose_list):
    """
    Fills missing (None) poses AND element-wise missing landmarks (0, 0, 0).
    1. Forward fills from previous frames.
    2. Backward fills for initial missing frames.
    3. Replaces entirely missing sequences with empty arrays.
    """
    n = len(pose_list)
    if n == 0:
        return pose_list

    # Helper function to create a boolean mask for (0, 0) landmarks
    # OpenPose outputs 0.0 for confidence as well when a joint is missing
    def is_zero_joint(pose):
        return (pose[:, 0] == 0.0) & (pose[:, 1] == 0.0)

    # Pass 1: Forward fill (take from previous frame)
    for i in range(1, n):
        if pose_list[i] is None:
            if pose_list[i-1] is not None:
                pose_list[i] = np.copy(pose_list[i-1])
        elif pose_list[i-1] is not None:
            # If both frames have detections, check for individual (0, 0) landmarks
            if pose_list[i].shape == pose_list[i-1].shape:
                missing_mask = is_zero_joint(pose_list[i])
                pose_list[i][missing_mask] = pose_list[i-1][missing_mask]

    # Pass 2: Backward fill (take from next frame to handle initial missing frames)
    for i in range(n - 2, -1, -1):
        if pose_list[i] is None:
            if pose_list[i+1] is not None:
                pose_list[i] = np.copy(pose_list[i+1])
        elif pose_list[i+1] is not None:
            # If both frames have detections, check for individual (0, 0) landmarks
            if pose_list[i].shape == pose_list[i+1].shape:
                missing_mask = is_zero_joint(pose_list[i])
                pose_list[i][missing_mask] = pose_list[i+1][missing_mask]

    # Pass 3: Fallback if a body part was NEVER detected in the whole sequence
    for i in range(n):
        if pose_list[i] is None:
            # UPDATED to 3 columns to account for confidence
            pose_list[i] = np.empty((0, 3))

    return pose_list

def process_directory(input_dir, output_npz_path, openpose_dir):
    print("Initializing OpenPose...")
    opWrapper = initialize_openpose(openpose_dir)
    
    img_paths_list = []
    body_poses_list = []
    left_hand_poses_list = []
    right_hand_poses_list = []

    valid_exts = ('.jpg', '.jpeg', '.png', '.bmp')
    files = [f for f in os.listdir(input_dir) if f.startswith('frame_') and f.lower().endswith(valid_exts)]
    files.sort()

    if not files:
        print(f"No 'frame_' images found in {input_dir}.")
        return

    print(f"Found {len(files)} frames. Starting extraction...")

    # Extraction Loop
    for file_name in files:
        image_path = os.path.join(input_dir, file_name)
        image_to_process = cv2.imread(image_path)
        
        if image_to_process is None:
            print(f"Warning: Could not read image at {image_path}. Skipping.")
            continue
            
        h, w, _ = image_to_process.shape

        datum = op.Datum()
        datum.cvInputData = image_to_process
        opWrapper.emplaceAndPop(op.VectorDatum([datum]))

        # Extract normalized poses (returns None if nothing detected)
        body_pose = extract_valid_pose(datum.poseKeypoints, w, h)
        
        left_hand_kp = datum.handKeypoints[0] if datum.handKeypoints is not None else None
        left_hand_pose = extract_valid_pose(left_hand_kp, w, h)
        
        right_hand_kp = datum.handKeypoints[1] if datum.handKeypoints is not None else None
        right_hand_pose = extract_valid_pose(right_hand_kp, w, h)

        # Track the raw extractions
        img_paths_list.append(file_name)
        body_poses_list.append(body_pose)
        left_hand_poses_list.append(left_hand_pose)
        right_hand_poses_list.append(right_hand_pose)

    print("Imputing missing keypoints using adjacent frames...")
    
    # Apply the forward/backward fill logic
    body_poses_list = fill_missing_poses(body_poses_list)
    left_hand_poses_list = fill_missing_poses(left_hand_poses_list)
    right_hand_poses_list = fill_missing_poses(right_hand_poses_list)

    # Save to npz
    np.savez(
        output_npz_path,
        img_path=np.array(img_paths_list),
        body_pose=np.array(body_poses_list, dtype=object),
        left_hand_pose=np.array(left_hand_poses_list, dtype=object),
        right_hand_pose=np.array(right_hand_poses_list, dtype=object)
    )
    print(f"Successfully processed {len(img_paths_list)} frames and saved to {output_npz_path}")


if __name__ == "__main__":
    # Define your directories and paths
    OPENPOSE_ROOT_DIR = "../src/openpose/"
    INPUT_DIR = sys.argv[1]
    OUTPUT_NPZ_FILE = sys.argv[2]

    try:
        process_directory(INPUT_DIR, OUTPUT_NPZ_FILE, OPENPOSE_ROOT_DIR)
    except Exception as e:
        print(f"An error occurred during extraction: {e}")
