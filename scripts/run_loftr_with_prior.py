"""
LoFTR + Detection Prior ROI filtering (Optimized for winter wheat repetitive texture).

Key optimizations:
  1. AND-filter: keep matches only when BOTH keypoints fall within ROIs (stricter filtering)
  2. Configurable ROI expansion: expand ROI by percentage to avoid over-tight cropping
  3. Configurable RANSAC threshold for homography estimation
  4. Configurable confidence threshold and top_k

Usage examples:
    # Test 5 pairs with default settings
    python scripts/run_loftr_with_prior.py --weights weights/outdoor_ds.ckpt --num_pairs 5

    # Experiment with different confidence thresholds
    python scripts/run_loftr_with_prior.py --weights weights/outdoor_ds.ckpt --conf_thr 0.3 --num_pairs 10
    python scripts/run_loftr_with_prior.py --weights weights/outdoor_ds.ckpt --conf_thr 0.5 --num_pairs 10

    # Adjust ROI expansion (15%) and RANSAC threshold (3.0)
    python scripts/run_loftr_with_prior.py --weights weights/outdoor_ds.ckpt --roi_expand 0.15 --ransac_thr 3.0

    # Run full dataset
    python scripts/run_loftr_with_prior.py --weights weights/outdoor_ds.ckpt
"""

import os
import sys
import argparse
import json
import cv2
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm


def load_prior_roi(prior_roi_dir, image_name):
    """
    Load prior ROI data for an image.
    Each ROI contains: roi=[x1, y1, x2, y2], score, class, roi_idx
    """
    json_path = Path(prior_roi_dir) / f"{Path(image_name).stem}.json"
    if not json_path.exists():
        return []
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data.get('objects', [])


def expand_roi(roi, expand_ratio, img_w, img_h):
    """
    Expand ROI by expand_ratio (e.g., 0.1 = 10%) in each direction,
    clamped to image boundaries.
    Args:
        roi: [x1, y1, x2, y2]
        expand_ratio: fraction to expand (e.g., 0.1 for 10%)
        img_w, img_h: image dimensions for clamping
    Returns:
        Expanded ROI [x1, y1, x2, y2]
    """
    x1, y1, x2, y2 = roi
    w = x2 - x1
    h = y2 - y1
    dx = w * expand_ratio
    dy = h * expand_ratio
    return [
        max(0, x1 - dx),
        max(0, y1 - dy),
        min(img_w, x2 + dx),
        min(img_h, y2 + dy),
    ]


def point_in_roi(point, roi):
    """Check if a point [x, y] is inside ROI [x1, y1, x2, y2]."""
    x, y = point
    x1, y1, x2, y2 = roi
    return x1 <= x <= x2 and y1 <= y <= y2


def filter_matches_by_roi(kpts0, kpts1, rois0, rois1):
    """
    AND-filter: keep matches only when BOTH keypoints fall within ANY ROI.
    This is stricter than OR-filter and better for repetitive textures
    because it ensures both matched points are on the target object.
    """
    boxes0 = [obj['roi'] for obj in rois0 if 'roi' in obj]
    boxes1 = [obj['roi'] for obj in rois1 if 'roi' in obj]

    # If no ROIs available, return all matches (no filtering)
    if not boxes0 or not boxes1:
        return np.ones(len(kpts0), dtype=bool)

    mask = np.zeros(len(kpts0), dtype=bool)
    for i in range(len(kpts0)):
        in_roi0 = any(point_in_roi(kpts0[i], box) for box in boxes0)
        in_roi1 = any(point_in_roi(kpts1[i], box) for box in boxes1)
        # AND-filter: both keypoints must be inside ROIs
        if in_roi0 and in_roi1:
            mask[i] = True
    return mask


def compute_homography(kpts0, kpts1, method=cv2.RANSAC, thr=4.0):
    """
    Compute homography matrix from matched keypoints using RANSAC.
    Args:
        thr: RANSAC reprojection threshold in pixels (smaller = stricter)
    """
    if len(kpts0) < 4:
        return None, []
    kpts0_np = np.float32(kpts0)
    kpts1_np = np.float32(kpts1)
    H, mask = cv2.findHomography(kpts0_np, kpts1_np, method, thr)
    inliers = mask.ravel().tolist() if mask is not None else []
    return H, inliers


def main():
    parser = argparse.ArgumentParser(
        description='LoFTR + Detection Prior ROI filtering (optimized for repetitive textures)')

    # Path arguments
    parser.add_argument('--images_dir', type=str,
                        default=r'E:\毕业材料\大论文\数据集\AutoX\images',
                        help='Directory containing input images')
    parser.add_argument('--prior_roi_dir', type=str,
                        default=r'E:\毕业材料\大论文\数据集\AutoX\prior_roi',
                        help='Directory containing prior ROI JSON files')
    parser.add_argument('--weights', type=str,
                        default=r'E:\workspacePython\LoFTR\weights\outdoor_ds.ckpt',
                        help='Path to LoFTR checkpoint')
    parser.add_argument('--output_dir', type=str,
                        default=r'E:\workspacePython\LoFTR\outputs\autox_loftr_prior',
                        help='Output directory for results')

    # Model arguments
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device: cuda or cpu')
    parser.add_argument('--resize', type=int, nargs='+', default=[1280],
                        help='Resize max dimension (default: 1280)')

    # Matching arguments
    parser.add_argument('--conf_thr', type=float, default=0.2,
                        help='Confidence threshold for LoFTR matches (建议: 0.2, 0.3, 0.4, 0.5)')
    parser.add_argument('--top_k', type=int, default=500,
                        help='Max number of matches to keep after confidence filtering (default: 500)')

    # ROI arguments
    parser.add_argument('--roi_expand', type=float, default=0.1,
                        help='ROI expansion ratio, e.g., 0.1 = 10%% expansion (default: 0.1)')

    # Homography arguments
    parser.add_argument('--ransac_thr', type=float, default=4.0,
                        help='RANSAC reprojection threshold in pixels (default: 4.0, 建议: 3.0~5.0)')

    # Execution arguments
    parser.add_argument('--num_pairs', type=int, default=0,
                        help='Number of pairs to process (0=all)')

    opt = parser.parse_args()

    # Auto-detect CUDA
    if opt.device == 'cuda' and not torch.cuda.is_available():
        print('CUDA not available, using CPU')
        opt.device = 'cpu'

    # Print config
    print('='*60)
    print('CONFIGURATION')
    print('='*60)
    print(f'  conf_thr:    {opt.conf_thr}')
    print(f'  top_k:       {opt.top_k}')
    print(f'  roi_expand:  {opt.roi_expand*100:.0f}%')
    print(f'  ransac_thr:  {opt.ransac_thr}')
    print(f'  resize:      {opt.resize[0]}')
    print(f'  device:      {opt.device}')
    print(f'  ROI filter:  AND (both keypoints must be in ROI)')
    print('='*60)

    Path(opt.output_dir).mkdir(parents=True, exist_ok=True)

    # Load LoFTR model
    print('\nLoading LoFTR model...')
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from src.loftr import LoFTR, default_cfg
    matcher = LoFTR(config=default_cfg)
    matcher.load_state_dict(torch.load(opt.weights, map_location='cpu', weights_only=True)['state_dict'])
    matcher = matcher.eval().to(device=opt.device)
    print('Model loaded!')

    # Get sorted image list
    image_dir = Path(opt.images_dir)
    image_files = sorted(set(
        list(image_dir.glob('*.jpg')) + list(image_dir.glob('*.png'))
        + list(image_dir.glob('*.JPG')) + list(image_dir.glob('*.JPEG'))
    ))
    print(f'Found {len(image_files)} images')

    # Determine number of pairs to process
    num_pairs = len(image_files) - 1
    if opt.num_pairs > 0:
        num_pairs = min(opt.num_pairs, len(image_files) - 1)
    print(f'Processing {num_pairs} pairs\n')

    # Results storage
    all_results = {
        'config': {
            'conf_thr': opt.conf_thr,
            'top_k': opt.top_k,
            'roi_expand': opt.roi_expand,
            'ransac_thr': opt.ransac_thr,
            'resize': opt.resize[0],
            'filter_mode': 'AND',
        },
        'images': [str(f.name) for f in image_files],
        'pairs': []
    }

    # Accumulate statistics
    stats = {
        'total_original_matches': 0,
        'total_filtered_matches': 0,
        'total_original_inliers': 0,
        'total_filtered_inliers': 0,
        'homo_original_success': 0,
        'homo_filtered_success': 0,
    }

    for i in tqdm(range(num_pairs), desc='Processing pairs'):
        img_path0 = image_files[i]
        img_path1 = image_files[i + 1]

        # Read images (np.fromfile supports Chinese paths)
        img0 = cv2.imdecode(np.fromfile(str(img_path0), dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
        img1 = cv2.imdecode(np.fromfile(str(img_path1), dtype=np.uint8), cv2.IMREAD_GRAYSCALE)

        if img0 is None or img1 is None:
            continue

        # Load prior ROIs for both images
        rois0 = load_prior_roi(opt.prior_roi_dir, img_path0.name)
        rois1 = load_prior_roi(opt.prior_roi_dir, img_path1.name)

        # Resize images (align to multiples of 32 for FPN compatibility)
        max_dim = max(img0.shape)
        scale = opt.resize[0] / max_dim
        new_w = (int(img0.shape[1] * scale) + 31) // 32 * 32
        new_h = (int(img0.shape[0] * scale) + 31) // 32 * 32
        img0_r = cv2.resize(img0, (new_w, new_h))
        img1_r = cv2.resize(img1, (new_w, new_h))

        # Scale ROI coordinates to resized image space
        scale_x = new_w / img0.shape[1]
        scale_y = new_h / img0.shape[0]
        for obj in rois0:
            if 'roi' in obj:
                obj['roi'] = [obj['roi'][0]*scale_x, obj['roi'][1]*scale_y,
                              obj['roi'][2]*scale_x, obj['roi'][3]*scale_y]
        for obj in rois1:
            if 'roi' in obj:
                obj['roi'] = [obj['roi'][0]*scale_x, obj['roi'][1]*scale_y,
                              obj['roi'][2]*scale_x, obj['roi'][3]*scale_y]

        # Expand ROIs to include surrounding context
        if opt.roi_expand > 0:
            for obj in rois0:
                if 'roi' in obj:
                    obj['roi'] = expand_roi(obj['roi'], opt.roi_expand, new_w, new_h)
            for obj in rois1:
                if 'roi' in obj:
                    obj['roi'] = expand_roi(obj['roi'], opt.roi_expand, new_w, new_h)

        # Run LoFTR inference
        batch = {
            'image0': torch.from_numpy(img0_r)[None][None].float().to(opt.device) / 255.,
            'image1': torch.from_numpy(img1_r)[None][None].float().to(opt.device) / 255.,
        }
        with torch.no_grad():
            matcher(batch)

        mkpts0 = batch['mkpts0_f'].cpu().numpy()
        mkpts1 = batch['mkpts1_f'].cpu().numpy()
        mconf = batch['mconf'].cpu().numpy()

        # Filter by confidence threshold
        conf_mask = mconf >= opt.conf_thr
        mkpts0 = mkpts0[conf_mask]
        mkpts1 = mkpts1[conf_mask]
        mconf = mconf[conf_mask]

        if len(mconf) == 0:
            continue

        # Keep top_k matches sorted by confidence (descending)
        sorted_idx = np.argsort(-mconf)[:opt.top_k]
        mkpts0 = mkpts0[sorted_idx]
        mkpts1 = mkpts1[sorted_idx]
        mconf = mconf[sorted_idx]

        total_matches = len(mkpts0)

        # === Baseline: original matches (no ROI filtering) ===
        H_orig, inliers_orig = compute_homography(mkpts0, mkpts1, thr=opt.ransac_thr)
        num_inliers_orig = sum(inliers_orig)

        # === ROI-filtered matches (AND-filter) ===
        roi_mask = filter_matches_by_roi(mkpts0, mkpts1, rois0, rois1)
        mkpts0_filtered = mkpts0[roi_mask]
        mkpts1_filtered = mkpts1[roi_mask]
        mconf_filtered = mconf[roi_mask]

        filtered_matches = len(mkpts0_filtered)
        H_filtered, inliers_filtered = compute_homography(
            mkpts0_filtered, mkpts1_filtered, thr=opt.ransac_thr)
        num_inliers_filtered = sum(inliers_filtered)

        # Accumulate statistics
        stats['total_original_matches'] += total_matches
        stats['total_filtered_matches'] += filtered_matches
        stats['total_original_inliers'] += num_inliers_orig
        stats['total_filtered_inliers'] += num_inliers_filtered
        if H_orig is not None:
            stats['homo_original_success'] += 1
        if H_filtered is not None:
            stats['homo_filtered_success'] += 1

        # Store per-pair results
        pair_result = {
            'pair': [img_path0.name, img_path1.name],
            'num_rois_img0': len([o for o in rois0 if 'roi' in o]),
            'num_rois_img1': len([o for o in rois1 if 'roi' in o]),
            'original': {
                'num_matches': total_matches,
                'num_inliers': num_inliers_orig,
                'inlier_ratio': num_inliers_orig / total_matches if total_matches > 0 else 0,
                'homography': H_orig.tolist() if H_orig is not None else None,
            },
            'filtered': {
                'num_matches': filtered_matches,
                'num_inliers': num_inliers_filtered,
                'inlier_ratio': num_inliers_filtered / filtered_matches if filtered_matches > 0 else 0,
                'homography': H_filtered.tolist() if H_filtered is not None else None,
            },
            'keypoints0': mkpts0.tolist(),
            'keypoints1': mkpts1.tolist(),
            'confidence': mconf.tolist(),
            'roi_mask': roi_mask.tolist(),
        }
        all_results['pairs'].append(pair_result)

        # Print per-pair result
        print(f'Pair {i}: {img_path0.name[:35]}...')
        print(f'  Original:  {total_matches} matches, {num_inliers_orig} inliers '
              f'({num_inliers_orig/total_matches*100:.1f}%), H: {"OK" if H_orig is not None else "FAIL"}')
        if filtered_matches > 0:
            print(f'  ROI-AND:   {filtered_matches} matches, {num_inliers_filtered} inliers '
                  f'({num_inliers_filtered/filtered_matches*100:.1f}%), H: {"OK" if H_filtered is not None else "FAIL"}')
        else:
            print(f'  ROI-AND:   0 matches (no keypoints in both ROIs)')

    # Save results to JSON
    output_file = Path(opt.output_dir) / 'loftr_prior_results.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2)

    # Print summary
    print('\n' + '='*60)
    print('SUMMARY')
    print('='*60)
    n = len(all_results['pairs'])
    if n > 0:
        print(f'Pairs processed:        {n}')
        print(f'')
        print(f'Original avg matches:   {stats["total_original_matches"]/n:.0f}')
        print(f'Filtered avg matches:   {stats["total_filtered_matches"]/n:.0f}')
        print(f'')
        print(f'Original avg inliers:   {stats["total_original_inliers"]/n:.1f}')
        print(f'Filtered avg inliers:   {stats["total_filtered_inliers"]/n:.1f}')
        print(f'')
        if stats["total_original_matches"] > 0:
            print(f'Original inlier ratio:  {stats["total_original_inliers"]/stats["total_original_matches"]*100:.1f}%')
        if stats["total_filtered_matches"] > 0:
            print(f'Filtered inlier ratio:  {stats["total_filtered_inliers"]/stats["total_filtered_matches"]*100:.1f}%')
        else:
            print(f'Filtered inlier ratio:  N/A (no matches passed AND-filter)')
        print(f'')
        print(f'Original H success:     {stats["homo_original_success"]}/{n} ({stats["homo_original_success"]/n*100:.1f}%)')
        print(f'Filtered H success:     {stats["homo_filtered_success"]}/{n} ({stats["homo_filtered_success"]/n*100:.1f}%)')
    print(f'\nResults saved to {output_file}')


if __name__ == '__main__':
    main()
