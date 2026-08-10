import os
import argparse
import cv2
import numpy as np
from pathlib import Path
from glob import glob
import mediapipe as mp

from CamSMPLifyX.utils.image_utils import crop, transform

# Detectron2 imports retained for bounding box generation
from core.constants import DETECTRON_CKPT, DETECTRON_CFG
from detectron2.config import LazyConfig
from core.utils.utils_detectron2 import DefaultPredictor_Lazy
import json

# ── MediaPipe Setup ────────────────────────────────────────────────────────────
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.framework.formats import landmark_pb2

base_options = python.BaseOptions(model_asset_path='data/hand_landmarker.task')
options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=2)
hand_detector = vision.HandLandmarker.create_from_options(options)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

IMG_RES = 768


def init_detector(threshold):
    """Initialize the Detectron2 object detector."""
    detectron2_cfg = LazyConfig.load(str(DETECTRON_CFG))
    detectron2_cfg.train.init_checkpoint = DETECTRON_CKPT

    for predictor in detectron2_cfg.model.roi_heads.box_predictors:
        predictor.test_score_thresh = threshold

    return DefaultPredictor_Lazy(detectron2_cfg)


def fill_blanks(estimation_data):
    """Interpolates missing keypoints for both 2D and 3D arrays."""
    n_frames = len(estimation_data['imgname'])
    keys_to_fill = ['mediapipe_kp_left', 'mediapipe_kp_right', 'mediapipe_kp_3d_left', 'mediapipe_kp_3d_right']

    for key in keys_to_fill:
        if key not in estimation_data:
            continue
            
        for i in range(1, n_frames - 1):
            if not np.any(estimation_data[key][i]):
                if np.any(estimation_data[key][i - 1]) and np.any(estimation_data[key][i + 1]):
                    estimation_data[key][i] = (
                        estimation_data[key][i - 1] + estimation_data[key][i + 1]
                    ) / 2


def select_best_hands(detection_result):
    """
    Returns the indices of the best left and right hands based on confidence score.
    Extracts both normalized 2D landmarks and 3D world landmarks.
    """
    best_idx = {'Left': -1, 'Right': -1}
    best_score = {'Left': -1.0, 'Right': -1.0}

    # Find the index of the best hand for each label
    for i, handedness in enumerate(detection_result.handedness):
        label = handedness[0].category_name  # 'Left' or 'Right'
        score = handedness[0].score

        if label in best_score and score > best_score[label]:
            best_score[label] = score
            best_idx[label] = i

    best_landmarks = {'Left': None, 'Right': None}
    best_world_landmarks = {'Left': None, 'Right': None}

    # Retrieve the landmarks using the best indices
    for label, idx in best_idx.items():
        if idx != -1:
            best_landmarks[label] = detection_result.hand_landmarks[idx]
            best_world_landmarks[label] = detection_result.hand_world_landmarks[idx]

    return best_landmarks, best_world_landmarks, best_score


def process_image(args, image_path, detector, output_folder, estimation_data):
    """Process a single image, extract MP keypoints inside Detectron bboxes, and save visualization."""
    img_cv2 = cv2.imread(str(image_path))
    if img_cv2 is None:
        print(f"Could not load {image_path}")
        return

    h_full, w_full, _ = img_cv2.shape

    # 1. Detectron2 Bounding Boxes
    det_out = detector(img_cv2)
    det_instances = det_out['instances']
    valid_idx = (det_instances.pred_classes == 0) & (det_instances.scores > args.detector_threshold)
    boxes = det_instances.pred_boxes.tensor[valid_idx].cpu().numpy()
    bbox_scale = (boxes[:, 2:4] - boxes[:, 0:2]) / 200.0
    bbox_center = (boxes[:, 2:4] + boxes[:, 0:2]) / 2

    # [Safety Catch] Ensure lists stay aligned with 'imgname' if no detections are found
    if len(boxes) == 0:
        print(f"No valid detections for {image_path}")
        estimation_data['mediapipe_kp_left'].append(np.zeros((21, 3)))
        estimation_data['mediapipe_kp_right'].append(np.zeros((21, 3)))
        estimation_data['mediapipe_kp_3d_left'].append(np.zeros((21, 3)))
        estimation_data['mediapipe_kp_3d_right'].append(np.zeros((21, 3)))
        return

    annotated_img = img_cv2.copy()
    crop_resized = None

    # 2. Iterate over detected persons
    for ind, box in enumerate(boxes):
        x1, y1, x2, y2 = map(int, box)

        crop_temp = img_cv2[y1:y2, x1:x2]
        if crop_temp.size == 0:
            continue

        crop_rgb = cv2.cvtColor(crop_temp, cv2.COLOR_BGR2RGB)
        h_crop, w_crop, _ = crop_rgb.shape
        crop_resized = crop(annotated_img, bbox_center[ind], bbox_scale[ind], [IMG_RES, IMG_RES]).astype(np.uint8)

        # 3. MediaPipe hand detection on the crop
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=crop_rgb)
        detection_result = hand_detector.detect(image)

        # 4. Resolve to (at most) one Left and one Right hand for this person
        best_landmarks, best_world_landmarks, best_score = select_best_hands(detection_result)

        person_hands_2d = {'Left': None, 'Right': None}
        person_hands_3d = {'Left': None, 'Right': None}
        
        for label in ['Left', 'Right']:
            # Handle 2D Landmarks
            if best_landmarks[label] is not None:
                hand_coords = [[lm.x * IMG_RES, lm.y * IMG_RES, lm.z] for lm in best_landmarks[label]]
                person_hands_2d[label] = np.array(hand_coords)

                hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
                hand_landmarks_proto.landmark.extend([
                    landmark_pb2.NormalizedLandmark(x=lm.x, y=lm.y, z=lm.z) for lm in best_landmarks[label]
                ])
                mp_drawing.draw_landmarks(
                    crop_resized,
                    hand_landmarks_proto,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style(),
                )
            
            # Handle 3D World Landmarks
            if best_world_landmarks[label] is not None:
                # world_landmarks are in metric scale, origin at hand center
                world_coords = [[lm.x, lm.y, lm.z] for lm in best_world_landmarks[label]]
                person_hands_3d[label] = np.array(world_coords)

        # 5. Append to tracking lists
        estimation_data['mediapipe_kp_left'].append(
            person_hands_2d['Left'] if person_hands_2d['Left'] is not None else np.zeros((21, 3))
        )
        estimation_data['mediapipe_kp_right'].append(
            person_hands_2d['Right'] if person_hands_2d['Right'] is not None else np.zeros((21, 3))
        )
        estimation_data['mediapipe_kp_3d_left'].append(
            person_hands_3d['Left'] if person_hands_3d['Left'] is not None else np.zeros((21, 3))
        )
        estimation_data['mediapipe_kp_3d_right'].append(
            person_hands_3d['Right'] if person_hands_3d['Right'] is not None else np.zeros((21, 3))
        )

    if crop_resized is not None:
        save_filename = os.path.join(output_folder, Path(image_path).name)
        cv2.imwrite(save_filename, crop_resized)
        print(f"Processed and saved: {save_filename}")


def main():
    parser = argparse.ArgumentParser(description='Detectron2 + MediaPipe Keypoints Extraction')
    parser.add_argument('--img_folder', type=str, default='demo_images', help='Input image folder')
    parser.add_argument('--out_folder', type=str, default='demo_out', help='Output folder')
    parser.add_argument('--detector_threshold', type=float, default=0.5, help='Detection threshold for Detectron2')
    parser.add_argument('--npz_file', type=str, default='demo_out_mediapipe.npz', help='Path to save keypoints in .npz format')
    parser.add_argument('--static', type=bool, default=False)

    args = parser.parse_args()

    # Initialize Detectron2
    detector = init_detector(args.detector_threshold)
    os.makedirs(args.out_folder, exist_ok=True)

    image_extensions = ('*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff', '*.webp')
    image_paths = [img for ext in image_extensions for img in glob(os.path.join(args.img_folder, ext))]
    image_paths = sorted(image_paths)

    # Dictionary to collect results for the .npz archive
    estimation_data = np.load(args.npz_file, allow_pickle=True)
    estimation_data = dict(estimation_data) if estimation_data is not None else {}
    estimation_data['mediapipe_kp_left'] = []
    estimation_data['mediapipe_kp_right'] = []
    estimation_data['mediapipe_kp_3d_left'] = []
    estimation_data['mediapipe_kp_3d_right'] = []

    for img_path in image_paths:
        process_image(args, img_path, detector, args.out_folder, estimation_data)
        
    fill_blanks(estimation_data)

    # Save extracted dictionary keypoints as an npz file
    np.savez(args.npz_file, **estimation_data)
    print(f"\nSaved keypoints array to {args.npz_file}")


if __name__ == '__main__':
    main()
