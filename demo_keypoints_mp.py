import os
import argparse
import cv2
import numpy as np
from pathlib import Path
from glob import glob
import mediapipe as mp

# Detectron2 imports retained for bounding box generation
from core.constants import DETECTRON_CKPT, DETECTRON_CFG
from detectron2.config import LazyConfig
from core.utils.utils_detectron2 import DefaultPredictor_Lazy
import json

# ── MediaPipe Setup ────────────────────────────────────────────────────────────
mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles


def init_detector(threshold):
    """Initialize the Detectron2 object detector."""
    detectron2_cfg = LazyConfig.load(str(DETECTRON_CFG))
    detectron2_cfg.train.init_checkpoint = DETECTRON_CKPT
    
    for predictor in detectron2_cfg.model.roi_heads.box_predictors:
        predictor.test_score_thresh = threshold
    
    return DefaultPredictor_Lazy(detectron2_cfg)


def process_image(args, image_path, detector, pose_model, hands_model, output_folder, estimation_data):
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

    if len(boxes) == 0:
        print(f"No valid detections for {image_path}")
        return

    annotated_img = img_cv2.copy()
    image_results = []

    # 2. Iterate over detected persons
    for ind, box in enumerate(boxes):
        x1, y1, x2, y2 = map(int, box)
        
        # Add a 10% margin to the bounding box to ensure full limbs/hands are captured
        margin_x = int((x2 - x1) * 0.1)
        margin_y = int((y2 - y1) * 0.1)
        x1, y1 = max(0, x1 - margin_x), max(0, y1 - margin_y)
        x2, y2 = min(w_full, x2 + margin_x), min(h_full, y2 + margin_y)
        
        crop = img_cv2[y1:y2, x1:x2]
        if crop.size == 0:
            continue
            
        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        h_crop, w_crop, _ = crop.shape

        # 3. MediaPipe Processing on the crop
        pose_results = pose_model.process(crop_rgb)
        hands_results = hands_model.process(crop_rgb)

        person_kps = {'bbox': [x1, y1, x2, y2], 'pose': None, 'hands': []}

        # ── Map and Draw POSE ──────────────────────────────────────────────────
        if pose_results.pose_landmarks:
            pose_coords = []
            for lm in pose_results.pose_landmarks.landmark:
                # Map back to absolute image coordinates for storage
                abs_x = lm.x * w_crop + x1
                abs_y = lm.y * h_crop + y1
                pose_coords.append([abs_x, abs_y, lm.z, lm.visibility])
                
                # Update landmark directly so mp_drawing plots correctly on the full image
                lm.x = abs_x / w_full
                lm.y = abs_y / h_full
                
            person_kps['pose'] = np.array(pose_coords)
            
            mp_drawing.draw_landmarks(
                annotated_img,
                pose_results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style(),
            )

        # ── Map and Draw HANDS ─────────────────────────────────────────────────
        if hands_results.multi_hand_landmarks:
            for hand_landmarks, handedness in zip(hands_results.multi_hand_landmarks, hands_results.multi_handedness):
                hand_coords = []
                for lm in hand_landmarks.landmark:
                    # Map back to absolute image coordinates for storage
                    abs_x = lm.x * w_crop + x1
                    abs_y = lm.y * h_crop + y1
                    hand_coords.append([abs_x, abs_y, lm.z])
                    
                    # Update landmark directly so mp_drawing plots correctly on the full image
                    lm.x = abs_x / w_full
                    lm.y = abs_y / h_full
                
                label = handedness.classification[0].label
                person_kps['hands'].append({
                    'label': label,
                    'keypoints': np.array(hand_coords)
                })

                # mp_drawing.draw_landmarks(
                #     annotated_img,
                #     hand_landmarks,
                #     mp_hands.HAND_CONNECTIONS,
                #     mp_drawing_styles.get_default_hand_landmarks_style(),
                #     mp_drawing_styles.get_default_hand_connections_style(),
                # )
        
        image_results.append(person_kps)
        # print(f"Person {ind}:")
        # print(json.dumps({
        #     'bbox': person_kps['bbox'],
        #     'pose': person_kps['pose'].tolist() if person_kps['pose'] is not None else None,
        #     'hands': [{'label': h['label'], 'keypoints': h['keypoints'].tolist()} for h in person_kps['hands']]
        # }, indent=2))
        for hand in person_kps['hands']:
            if hand['label'] == 'Left':
                estimation_data['mediapipe_kp_left'].append(hand['keypoints'])
            else:
                estimation_data['mediapipe_kp_right'].append(hand['keypoints'])

    # save_filename = os.path.join(output_folder, Path(image_path).name)
    # cv2.imwrite(save_filename, annotated_img)
    # print(f"Processed and saved: {save_filename}")


def main():
    parser = argparse.ArgumentParser(description='Detectron2 + MediaPipe Keypoints Extraction')
    parser.add_argument('--img_folder', type=str, default='demo_images', help='Input image folder')
    parser.add_argument('--out_folder', type=str, default='demo_out', help='Output folder')
    parser.add_argument('--detector_threshold', type=float, default=0.5, help='Detection threshold for Detectron2')
    parser.add_argument('--npz_file', type=str, default='demo_out_mediapipe.npz', help='Path to save keypoints in .npz format')

    args = parser.parse_args()

    # Initialize Detectron2
    detector = init_detector(args.detector_threshold)
    os.makedirs(args.out_folder, exist_ok=True)
    
    # Initialize MediaPipe instances
    pose_model = mp_pose.Pose(static_image_mode=True, model_complexity=2)
    hands_model = mp_hands.Hands(static_image_mode=True, max_num_hands=2)

    image_extensions = ('*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff', '*.webp')
    image_paths = [img for ext in image_extensions for img in glob(os.path.join(args.img_folder, ext))]
    image_paths = sorted(image_paths) 
    
    # Dictionary to collect results for the .npz archive
    estimation_data = np.load(args.npz_file, allow_pickle=True)
    estimation_data = dict(estimation_data) if estimation_data is not None else {}
    estimation_data['mediapipe_kp_left'] = []
    estimation_data['mediapipe_kp_right'] = []
    
    for img_path in image_paths:
        process_image(args, img_path, detector, pose_model, hands_model, args.out_folder, estimation_data)
    
    # Save extracted dictionary keypoints as an npz file
    np.savez(args.npz_file, **estimation_data)
    print(f"\nSaved keypoints array to {args.npz_file}")


if __name__ == '__main__':
    main()