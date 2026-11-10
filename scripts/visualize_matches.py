"""
Visualize LoFTR matching results.
Draws matched keypoints as lines between image pairs.

Usage:
    # Visualize baseline results
    python scripts/visualize_matches.py --input outputs/autox_loftr/loftr_results.json --num_pairs 10

    # Visualize ROI-crop results
    python scripts/visualize_matches.py --input outputs/autox_loftr_roi_crop/loftr_roi_crop_results.json --num_pairs 10

    # Visualize with prior filtering
    python scripts/visualize_matches.py --input outputs/autox_loftr_prior/loftr_prior_results.json --num_pairs 10
"""

import os
import sys
import argparse
import json
import cv2
import numpy as np
from pathlib import Path


def draw_matches(img0, img1, kpts0, kpts1, conf, inlier_mask=None, max_lines=200):
    """
    Draw matching lines between two images side by side.
    Args:
        img0, img1: grayscale images
        kpts0, kpts1: matched keypoint coordinates (N, 2)
        conf: confidence scores (N,)
        inlier_mask: boolean mask for inliers (N,), None = show all
        max_lines: max number of lines to draw
    """
    h0, w0 = img0.shape[:2]
    h1, w1 = img1.shape[:2]

    # Make both images same height
    h = max(h0, h1)
    if h0 != h:
        img0 = cv2.resize(img0, (int(w0 * h / h0), h))
    if h1 != h:
        img1 = cv2.resize(img1, (int(w1 * h / h1), h))

    # Create side-by-side canvas
    gap = 10
    canvas = np.ones((h, img0.shape[1] + gap + img1.shape[1], 3), dtype=np.uint8) * 255
    canvas[:, :img0.shape[1]] = cv2.cvtColor(img0, cv2.COLOR_GRAY2BGR) if len(img0.shape) == 2 else img0
    canvas[:, img0.shape[1]+gap:] = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR) if len(img1.shape) == 2 else img1

    # Scale keypoints to canvas
    sx0 = img0.shape[1] / w0
    sy0 = img0.shape[0] / h0
    sx1 = img1.shape[1] / w1
    sy1 = img1.shape[0] / h1

    # Limit lines and sort by confidence
    n = min(len(kpts0), max_lines)
    if n == 0:
        return canvas

    idx = np.argsort(-conf)[:n]

    # Color by inlier/outlier
    for i in idx:
        pt0 = (int(kpts0[i][0] * sx0), int(kpts0[i][1] * sy0))
        pt1 = (int(kpts1[i][0] * sx1 + img0.shape[1] + gap), int(kpts1[i][1] * sy1))

        if inlier_mask is not None:
            color = (0, 200, 0) if inlier_mask[i] else (100, 100, 100)  # green=inlier, gray=outlier
        else:
            # Color by confidence: high=red, low=blue
            c_val = conf[i]
            color = (0, int(255 * c_val), int(255 * (1 - c_val)))

        cv2.line(canvas, pt0, pt1, color, 1, cv2.LINE_AA)
        cv2.circle(canvas, pt0, 3, color, -1)
        cv2.circle(canvas, pt1, 3, color, -1)

    # Legend
    cv2.putText(canvas, f'{n} matches shown', (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    if inlier_mask is not None:
        n_inliers = sum(inlier_mask[idx])
        cv2.putText(canvas, f'{int(n_inliers)} inliers (green)', (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 0), 2)

    return canvas


def main():
    parser = argparse.ArgumentParser(description='Visualize LoFTR matching results')
    parser.add_argument('--input', type=str, required=True,
                        help='Path to results JSON file')
    parser.add_argument('--images_dir', type=str, default=None,
                        help='Image directory (auto-detected from JSON if not set)')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory (default: <input_dir>/vis)')
    parser.add_argument('--num_pairs', type=int, default=10,
                        help='Number of pairs to visualize (default: 10)')
    parser.add_argument('--max_lines', type=int, default=200,
                        help='Max match lines to draw (default: 200)')

    opt = parser.parse_args()

    # Load results
    with open(opt.input, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Setup output directory
    if opt.output_dir is None:
        opt.output_dir = str(Path(opt.input).parent / 'vis')
    Path(opt.output_dir).mkdir(parents=True, exist_ok=True)

    # Detect image directory
    if opt.images_dir is None:
        # Try to find from first pair
        pairs = data.get('pairs', data.get('matches', []))
        if pairs:
            first_img = pairs[0]['pair'][0]
            # Check common directories
            for candidate in [
                r'E:\毕业材料\大论文\数据集\AutoX\images',
                r'E:\毕业材料\大论文\数据集\AutoX\visualization',
            ]:
                if Path(candidate).joinpath(first_img).exists():
                    opt.images_dir = candidate
                    break

    if opt.images_dir is None:
        print('ERROR: Cannot find images directory. Please specify --images_dir')
        return

    print(f'Image dir: {opt.images_dir}')
    print(f'Output dir: {opt.output_dir}')

    # Determine result format
    pairs_data = data.get('pairs', data.get('matches', []))
    num_pairs = min(opt.num_pairs, len(pairs_data))

    for i in range(num_pairs):
        pair = pairs_data[i]
        img_name0, img_name1 = pair['pair']

        # Read images
        img_path0 = Path(opt.images_dir) / img_name0
        img_path1 = Path(opt.images_dir) / img_name1
        img0 = cv2.imdecode(np.fromfile(str(img_path0), dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
        img1 = cv2.imdecode(np.fromfile(str(img_path1), dtype=np.uint8), cv2.IMREAD_GRAYSCALE)

        if img0 is None or img1 is None:
            print(f'Skipping pair {i}: cannot read images')
            continue

        # Extract keypoints and confidence
        # Handle different JSON formats
        if 'keypoints0' in pair:
            kpts0 = np.array(pair['keypoints0'])
            kpts1 = np.array(pair['keypoints1'])
            conf = np.array(pair['confidence'])
        elif 'original' in pair:
            # run_loftr_roi_crop format - use roi_crop keypoints
            kpts0 = np.array(pair.get('keypoints0', []))
            kpts1 = np.array(pair.get('keypoints1', []))
            conf = np.array(pair.get('confidence', []))
        else:
            print(f'Skipping pair {i}: unknown format')
            continue

        if len(kpts0) == 0:
            print(f'Skipping pair {i}: no matches')
            continue

        # Get inlier mask from homography if available
        inlier_mask = None
        homo_data = pair.get('roi_crop', pair.get('filtered', pair))
        if homo_data and 'homography' in homo_data and homo_data['homography'] is not None:
            # Reconstruct inlier mask from homography
            H = np.array(homo_data['homography'])
            pts0_h = np.hstack([kpts0, np.ones((len(kpts0), 1))])
            pts1_proj = (H @ pts0_h.T).T
            pts1_proj = pts1_proj[:, :2] / pts1_proj[:, 2:3]
            errors = np.linalg.norm(pts1_proj - kpts1, axis=1)
            ransac_thr = data.get('config', {}).get('ransac_thr', 4.0)
            inlier_mask = errors < ransac_thr

        # Draw matches
        canvas = draw_matches(img0, img1, kpts0, kpts1, conf, inlier_mask, opt.max_lines)

        # Add title
        title = f'Pair {i}: {img_name0[:25]}... <-> {img_name1[:25]}...'
        cv2.putText(canvas, title, (10, canvas.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        # Save
        out_path = Path(opt.output_dir) / f'vis_{i:04d}.png'
        cv2.imwrite(str(out_path), canvas)
        print(f'Saved: {out_path.name} ({len(kpts0)} matches)')

    print(f'\nDone! {num_pairs} visualizations saved to {opt.output_dir}')


if __name__ == '__main__':
    main()
