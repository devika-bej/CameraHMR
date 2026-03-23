import os
import torch
import argparse
import cv2
import numpy as np
from pathlib import Path
from glob import glob
from collections import OrderedDict
from matplotlib import pyplot as plt

from core.utils import recursive_to
from core.datasets.dataset import Dataset
from core.densekp_trainer import DenseKP
from core.utils.renderer_pyrd import Renderer
from core.utils.train_utils import denormalize_images
from core.constants import DETECTRON_CKPT, DETECTRON_CFG, DENSEKP_CKPT
from detectron2.config import LazyConfig
from core.utils.utils_detectron2 import DefaultPredictor_Lazy
import matplotlib.colors as mcolors
import matplotlib.cm as cm


def init_detector(threshold):
    """Initialize the Detectron2 object detector."""
    detectron2_cfg = LazyConfig.load(str(DETECTRON_CFG))
    detectron2_cfg.train.init_checkpoint = DETECTRON_CKPT
    
    for predictor in detectron2_cfg.model.roi_heads.box_predictors:
        predictor.test_score_thresh = threshold
    
    return DefaultPredictor_Lazy(detectron2_cfg)


def process_image(args, image_path, model, detector, device, output_folder, estimation_data=None):
    """Process a single image and save the output visualization."""
    img_cv2 = cv2.imread(str(image_path))
    det_out = detector(img_cv2)
    
    det_instances = det_out['instances']
    valid_idx = (det_instances.pred_classes == 0) & (det_instances.scores > 0.5)
    boxes = det_instances.pred_boxes.tensor[valid_idx].cpu().numpy()

    if len(boxes) == 0:
        print(f"No valid detections for {image_path}")
        return
    
    bbox_scale = (boxes[:, 2:4] - boxes[:, 0:2]) / 200.0
    bbox_center = (boxes[:, 2:4] + boxes[:, 0:2]) / 2.0
    
    dataset = Dataset(img_cv2, bbox_center, bbox_scale, None, False, image_path)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    for batch in dataloader:
        batch = recursive_to(batch, device)
        with torch.no_grad():
            out = model(batch)
        denormalized_images = denormalize_images(batch['img'])
        for ind, img in enumerate(denormalized_images):
            img = img.detach().cpu().numpy().transpose(1, 2, 0) * 255
            img = np.clip(img, 0, 255).astype(np.uint8)

            fig, ax = plt.subplots(dpi=300)
            ax.imshow(img)

            pred_vertices = out['pred_keypoints'][ind].detach().cpu().numpy()
            
            estimation_data['dense_kp'].append(pred_vertices)
            
            pred_vertices = (pred_vertices + 0.5) * 256
            confidences = pred_vertices[:, 2]
        
            norm = mcolors.Normalize(vmin=args.vmin, vmax=args.vmax) # Based on samples average
        
            # Get colors based on confidence using a colormap (e.g., 'viridis')
            colors = cm.viridis(norm(confidences))  

            # Scatter with colors mapped from confidences
            ax.scatter(pred_vertices[:, 0], pred_vertices[:, 1], s=15, c=colors, edgecolors='black', linewidth=0.5)

            save_filename = os.path.join(output_folder, Path(image_path).stem + f"_{ind}.png")
            plt.savefig(save_filename)
            plt.close()

def main():
    parser = argparse.ArgumentParser(description='DenseKP demo')
    parser.add_argument('--checkpoint', type=str, default=DENSEKP_CKPT, help='Path to pretrained model checkpoint')
    parser.add_argument('--img_folder', type=str, default='demo_images', help='Input image folder')
    parser.add_argument('--out_folder', type=str, default='demo_out', help='Output folder')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for inference')
    parser.add_argument('--num_workers', type=int, default=1, help='Num of workers')
    parser.add_argument('--detector_threshold', type=float, default=0.25, help='Detection threshold')
    parser.add_argument('--vmin', type=float, default=-500., help='Minimum possible sigma value for the checkpoint')
    parser.add_argument('--vmax', type=float, default=1500., help='Maximum possible sigma value for the checkpoint')
    parser.add_argument('--npz_file', type=str, default='demo_out', help='Path to save keypoints in .npz format')
    # vmin and vmax depends on the checkpoint and how long the model is trained

    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = DenseKP.load_from_checkpoint(args.checkpoint, strict=False).to(device).eval()
    detector = init_detector(args.detector_threshold)
    os.makedirs(args.out_folder, exist_ok=True)

    image_extensions = ('*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff', '*.webp')
    image_paths = [img for ext in image_extensions for img in glob(os.path.join(args.img_folder, ext))]
    image_paths = sorted(image_paths)  # Sort for consistent processing order
    
    estimation_data = np.load(args.npz_file) if os.path.exists(args.npz_file) else None
    estimation_data = dict(estimation_data) if estimation_data is not None else None
    estimation_data['dense_kp'] = []  # Initialize list to store dense keypoints
    
    for ind, img_path in enumerate(image_paths):
        process_image(args, img_path, model, detector, device, args.out_folder, estimation_data)
    
    np.savez(args.npz_file, **estimation_data)


if __name__ == '__main__':
    main()
