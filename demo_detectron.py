import os
import argparse
import cv2
import numpy as np
from pathlib import Path
from glob import glob

from CamSMPLifyX.utils.image_utils import crop, transform

# Detectron2 imports retained for bounding box generation
from core.constants import DETECTRON_CKPT, DETECTRON_CFG
from detectron2.config import LazyConfig
from core.utils.utils_detectron2 import DefaultPredictor_Lazy
import json

IMG_RES = 768


def init_detector(threshold):
    """Initialize the Detectron2 object detector."""
    detectron2_cfg = LazyConfig.load(str(DETECTRON_CFG))
    detectron2_cfg.train.init_checkpoint = DETECTRON_CKPT
    
    for predictor in detectron2_cfg.model.roi_heads.box_predictors:
        predictor.test_score_thresh = threshold
    
    return DefaultPredictor_Lazy(detectron2_cfg)


def process_image(args, image_path, detector, output_folder):
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

    if len(boxes) == 0:
        print(f"No valid detections for {image_path}")
        return

    annotated_img = img_cv2.copy()
    image_results = []

    # 2. Iterate over detected persons
    left_over_left = 0
    left_over_right = 0
    for ind, box in enumerate(boxes):
        x1, y1, x2, y2 = map(int, box)
        
        crop_temp = img_cv2[y1:y2, x1:x2]
        if crop_temp.size == 0:
            continue
            
        crop_rgb = cv2.cvtColor(crop_temp, cv2.COLOR_BGR2RGB)
        h_crop, w_crop, _ = crop_rgb.shape
        crop_resized = crop(annotated_img, bbox_center[ind], bbox_scale[ind], [IMG_RES, IMG_RES]).astype(np.uint8)

    save_filename = os.path.join(output_folder, f"resized_{Path(image_path).name}")
    cv2.imwrite(save_filename, crop_resized)
    save_filename = os.path.join(output_folder, Path(image_path).name)
    cv2.imwrite(save_filename, crop_rgb)
    print(f"Processed and saved: {save_filename}")


def main():
    parser = argparse.ArgumentParser(description='Detectron2 + MediaPipe Keypoints Extraction')
    parser.add_argument('--img_folder', type=str, default='demo_images', help='Input image folder')
    parser.add_argument('--out_folder', type=str, default='demo_out', help='Output folder')
    parser.add_argument('--detector_threshold', type=float, default=0.5, help='Detection threshold for Detectron2')
#    parser.add_argument('--npz_file', type=str, default='demo_out_mediapipe.npz', help='Path to save keypoints in .npz format')
    parser.add_argument('--static', type=bool, default=False)

    args = parser.parse_args()

    # Initialize Detectron2
    detector = init_detector(args.detector_threshold)
    os.makedirs(args.out_folder, exist_ok=True)

    image_extensions = ('*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff', '*.webp')
    image_paths = [img for ext in image_extensions for img in glob(os.path.join(args.img_folder, ext))]
    image_paths = sorted(image_paths)
    
    for img_path in image_paths:
        process_image(args, img_path, detector, args.out_folder)


if __name__ == '__main__':
    main()
