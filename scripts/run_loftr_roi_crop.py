"""
LoFTR + Detection Prior: Per-ROI Crop Matching (Optimized).

Pipeline:
  1. Read original image pair
  2. Load ROIs from prior_roi JSON, validate roi_idx consistency
  3. For each ROI pair (matched by roi_idx):
     - Crop ROI region from both images
     - Maintain aspect ratio, pad to crop_size x crop_size
     - Run LoFTR on the padded crop
     - Per-ROI RANSAC to remove outliers within this ROI
     - Keep only ROI-internal inliers (high quality matches)
     - Map keypoints back to original image coordinates
  4. Merge all ROI inliers
  5. Global confidence sort, limit to max_total_matches
  6. Final RANSAC homography on merged matches

Usage:
    # Test 5 pairs (baseline uses same resize=1024 as run_loftr_autox.py)
    python scripts/run_loftr_roi_crop.py --weights weights/outdoor_ds.ckpt --num_pairs 5

    # Experiment with different conf_thr
    python scripts/run_loftr_roi_crop.py --weights weights/outdoor_ds.ckpt --conf_thr 0.4 --num_pairs 10
    python scripts/run_loftr_roi_crop.py --weights weights/outdoor_ds.ckpt --conf_thr 0.5 --num_pairs 10

    # Ablation: roi_expand 0.15
    python scripts/run_loftr_roi_crop.py --weights weights/outdoor_ds.ckpt --roi_expand 0.15 --num_pairs 10

    # Ablation: tighter roi_ransac_thr
    python scripts/run_loftr_roi_crop.py --weights weights/outdoor_ds.ckpt --roi_ransac_thr 3.0 --num_pairs 10

    # Full dataset
    python scripts/run_loftr_roi_crop.py --weights weights/outdoor_ds.ckpt
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
    Load prior ROI data, return dict keyed by roi_idx.
    roi_idx is supposed to represent the same target across consecutive frames.
    """
    json_path = Path(prior_roi_dir) / f"{Path(image_name).stem}.json"
    if not json_path.exists():
        return {}
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    roi_dict = {}
    for obj in data.get('objects', []):
        if 'roi' in obj and 'roi_idx' in obj:
            roi_dict[obj['roi_idx']] = obj
    return roi_dict


def expand_roi(roi, expand_ratio, img_w, img_h):
    """Expand ROI by ratio, clamped to image boundaries."""
    x1, y1, x2, y2 = roi
    w, h = x2 - x1, y2 - y1
    dx, dy = w * expand_ratio, h * expand_ratio
    return [
        max(0, x1 - dx), max(0, y1 - dy),
        min(img_w, x2 + dx), min(img_h, y2 + dy),
    ]


def crop_and_pad(img, roi, crop_size=640):
    """
    Crop ROI from image, maintain aspect ratio, pad to crop_size x crop_size.
    Returns:
        padded: crop_size x crop_size image (padded with zeros if needed)
        offset: (x1, y1) offset of ROI in original image
        scale: resize scale factor (for mapping back coords)
        roi_size: (w_orig, h_orig) original ROI dimensions
    """
    x1, y1, x2, y2 = [int(round(v)) for v in roi]
    h_orig = y2 - y1
    w_orig = x2 - x1

    if w_orig <= 0 or h_orig <= 0:
        return None, None, None, None

    # Crop from original image
    crop = img[y1:y2, x1:x2]

    # Maintain aspect ratio: fit into crop_size x crop_size
    scale = crop_size / max(w_orig, h_orig)
    new_w = int(w_orig * scale)
    new_h = int(h_orig * scale)
    crop_resized = cv2.resize(crop, (new_w, new_h))

    # Pad to crop_size x crop_size (bottom-right padding)
    padded = np.zeros((crop_size, crop_size), dtype=crop.dtype)
    padded[:new_h, :new_w] = crop_resized

    return padded, (x1, y1), scale, (w_orig, h_orig)


def compute_homography(kpts0, kpts1, method=cv2.RANSAC, thr=4.0):
    """Compute homography matrix from matched keypoints."""
    if len(kpts0) < 4:
        return None, []
    H, mask = cv2.findHomography(
        np.float32(kpts0), np.float32(kpts1), method, thr)
    inliers = mask.ravel().tolist() if mask is not None else []
    return H, inliers


def main():
    parser = argparse.ArgumentParser(
        description='LoFTR + Detection Prior: Per-ROI Crop Matching (Optimized)')

    # Path arguments
    parser.add_argument('--images_dir', type=str,
                        default=r'E:\毕业材料\大论文\数据集\AutoX\images')
    parser.add_argument('--prior_roi_dir', type=str,
                        default=r'E:\毕业材料\大论文\数据集\AutoX\prior_roi')
    parser.add_argument('--weights', type=str,
                        default=r'E:\workspacePython\LoFTR\weights\outdoor_ds.ckpt')
    parser.add_argument('--output_dir', type=str,
                        default=r'E:\workspacePython\LoFTR\outputs\autox_loftr_roi_crop')

    # Model arguments
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--resize', type=int, nargs='+', default=[1024],
                        help='Resize max dimension for full-image baseline (default: 1024, same as run_loftr_autox.py)')
    parser.add_argument('--crop_size', type=int, default=1024,
                        help='Crop padded size (default: 640)')

    # Matching arguments
    parser.add_argument('--conf_thr', type=float, default=0.4,
                        help='Confidence threshold (default: 0.4)')
    parser.add_argument('--top_k_per_roi', type=int, default=50,
                        help='Max matches to keep per ROI after LoFTR (default: 50)')
    parser.add_argument('--max_total_matches', type=int, default=1000,
                        help='Max total matches after global merge (default: 1000)')

    # ROI arguments
    parser.add_argument('--roi_expand', type=float, default=0.1,
                        help='ROI expansion ratio (default: 0.1 = 10%%, ablation: 0.15)')

    # Homography arguments
    parser.add_argument('--ransac_thr', type=float, default=4.0,
                        help='RANSAC reprojection threshold (default: 4.0)')
    parser.add_argument('--roi_ransac_thr', type=float, default=5.0,
                        help='Per-ROI RANSAC threshold (default: 5.0, ablation: 3.0 or 4.0)')

    # Execution
    parser.add_argument('--num_pairs', type=int, default=0,
                        help='Number of pairs to process (0=all)')
    parser.add_argument('--skip_baseline', action='store_true',
                        help='Skip full-image baseline (faster)')

    opt = parser.parse_args()

    if opt.device == 'cuda' and not torch.cuda.is_available():
        print('CUDA not available, using CPU')
        opt.device = 'cpu'

    # Print config
    print('='*60)
    print('CONFIGURATION')
    print('='*60)
    print(f'  resize (baseline): {opt.resize[0]} (same as run_loftr_autox.py)')
    print(f'  crop_size:         {opt.crop_size}x{opt.crop_size} (aspect-ratio preserved + pad)')
    print(f'  conf_thr:          {opt.conf_thr}')
    print(f'  top_k_per_roi:     {opt.top_k_per_roi}')
    print(f'  max_total_matches: {opt.max_total_matches}')
    print(f'  roi_expand:        {opt.roi_expand*100:.0f}%')
    print(f'  ransac_thr:        {opt.ransac_thr}')
    print(f'  roi_ransac_thr:    {opt.roi_ransac_thr}')
    print(f'  device:            {opt.device}')
    print(f'  pipeline:          LoFTR -> per-ROI RANSAC -> merge -> global sort -> final RANSAC')
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

    num_pairs = len(image_files) - 1
    if opt.num_pairs > 0:
        num_pairs = min(opt.num_pairs, len(image_files) - 1)
    print(f'Processing {num_pairs} pairs\n')

    # Results storage
    all_results = {
        'config': {
            'resize': opt.resize[0],
            'crop_size': opt.crop_size,
            'conf_thr': opt.conf_thr,
            'top_k_per_roi': opt.top_k_per_roi,
            'max_total_matches': opt.max_total_matches,
            'roi_expand': opt.roi_expand,
            'ransac_thr': opt.ransac_thr,
            'roi_ransac_thr': opt.roi_ransac_thr,
            'pipeline': 'per_roi_crop_with_internal_ransac',
        },
        'images': [str(f.name) for f in image_files],
        'pairs': []
    }

    stats = {
        'total_rois': 0,
        'total_original_matches': 0,
        'total_roi_matches': 0,
        'total_original_inliers': 0,
        'total_roi_inliers': 0,
        'homo_original_success': 0,
        'homo_roi_success': 0,
    }

    for i in tqdm(range(num_pairs), desc='Processing pairs'):
        img_path0 = image_files[i]
        img_path1 = image_files[i + 1]

        # Read original images
        img0 = cv2.imdecode(np.fromfile(str(img_path0), dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
        img1 = cv2.imdecode(np.fromfile(str(img_path1), dtype=np.uint8), cv2.IMREAD_GRAYSCALE)

        if img0 is None or img1 is None:
            continue

        h0, w0 = img0.shape
        h1, w1 = img1.shape

        # Load ROIs (keyed by roi_idx)
        rois0 = load_prior_roi(opt.prior_roi_dir, img_path0.name)
        rois1 = load_prior_roi(opt.prior_roi_dir, img_path1.name)

        # Find common ROI indices
        common_roi_ids = sorted(set(rois0.keys()) & set(rois1.keys()))
        num_rois = len(common_roi_ids)

        # Warn if no common ROIs
        if num_rois == 0:
            print(f'Pair {i}: No common ROIs between images, skipping')
            continue

        # Warn about roi_idx consistency (informational only)
        roi_ids0 = set(rois0.keys())
        roi_ids1 = set(rois1.keys())
        only_in_0 = roi_ids0 - roi_ids1
        only_in_1 = roi_ids1 - roi_ids0
        if only_in_0 or only_in_1:
            print(f'  Note: roi_idx mismatch — only in img0: {len(only_in_0)}, only in img1: {len(only_in_1)}')

        # === Baseline: full-image LoFTR (same preprocessing as run_loftr_autox.py) ===
        total_original = 0
        num_inliers_orig = 0
        H_orig = None
        if not opt.skip_baseline:
            max_dim = max(h0, w0, h1, w1)
            scale_full = opt.resize[0] / max_dim
            new_w = (int(w0 * scale_full) + 31) // 32 * 32
            new_h = (int(h0 * scale_full) + 31) // 32 * 32
            img0_full = cv2.resize(img0, (new_w, new_h))
            img1_full = cv2.resize(img1, (new_w, new_h))

            batch_full = {
                'image0': torch.from_numpy(img0_full)[None][None].float().to(opt.device) / 255.,
                'image1': torch.from_numpy(img1_full)[None][None].float().to(opt.device) / 255.,
            }
            with torch.no_grad():
                matcher(batch_full)

            mkpts0_full = batch_full['mkpts0_f'].cpu().numpy()
            mkpts1_full = batch_full['mkpts1_f'].cpu().numpy()
            mconf_full = batch_full['mconf'].cpu().numpy()

            conf_mask = mconf_full >= opt.conf_thr
            mkpts0_full = mkpts0_full[conf_mask]
            mkpts1_full = mkpts1_full[conf_mask]
            mconf_full = mconf_full[conf_mask]
            if len(mconf_full) > 0:
                idx = np.argsort(-mconf_full)[:opt.top_k_per_roi * 10]
                mkpts0_full, mkpts1_full, mconf_full = mkpts0_full[idx], mkpts1_full[idx], mconf_full[idx]

            mkpts0_full_orig = mkpts0_full / scale_full
            mkpts1_full_orig = mkpts1_full / scale_full

            total_original = len(mkpts0_full_orig)
            H_orig, inliers_orig = compute_homography(mkpts0_full_orig, mkpts1_full_orig, thr=opt.ransac_thr)
            num_inliers_orig = sum(inliers_orig)

        # === Per-ROI crop matching with internal RANSAC ===
        roi_all_kpts0 = []
        roi_all_kpts1 = []
        roi_all_conf = []
        roi_stats = []  # per-ROI statistics

        for roi_idx in common_roi_ids:
            roi_obj0 = rois0[roi_idx]
            roi_obj1 = rois1[roi_idx]

            # Expand ROI
            roi0_exp = expand_roi(roi_obj0['roi'], opt.roi_expand, w0, h0)
            roi1_exp = expand_roi(roi_obj1['roi'], opt.roi_expand, w1, h1)

            # Crop, maintain aspect ratio, pad to crop_size x crop_size
            crop0, off0, sc0, sz0 = crop_and_pad(img0, roi0_exp, opt.crop_size)
            crop1, off1, sc1, sz1 = crop_and_pad(img1, roi1_exp, opt.crop_size)

            if crop0 is None or crop1 is None:
                continue

            # Run LoFTR on padded crops
            batch = {
                'image0': torch.from_numpy(crop0)[None][None].float().to(opt.device) / 255.,
                'image1': torch.from_numpy(crop1)[None][None].float().to(opt.device) / 255.,
            }
            with torch.no_grad():
                matcher(batch)

            kpts0_crop = batch['mkpts0_f'].cpu().numpy()
            kpts1_crop = batch['mkpts1_f'].cpu().numpy()
            conf = batch['mconf'].cpu().numpy()

            # Filter by confidence
            c_mask = conf >= opt.conf_thr
            kpts0_crop = kpts0_crop[c_mask]
            kpts1_crop = kpts1_crop[c_mask]
            conf = conf[c_mask]

            if len(conf) == 0:
                roi_stats.append({'roi_idx': roi_idx, 'raw_matches': 0, 'after_ransac': 0})
                continue

            # Keep top_k_per_roi
            if len(conf) > opt.top_k_per_roi:
                idx = np.argsort(-conf)[:opt.top_k_per_roi]
                kpts0_crop, kpts1_crop, conf = kpts0_crop[idx], kpts1_crop[idx], conf[idx]

            raw_count = len(conf)

            # Map crop keypoints back to original image coordinates
            kpts0_orig = kpts0_crop / sc0 + np.array([off0[0], off0[1]])
            kpts1_orig = kpts1_crop / sc1 + np.array([off1[0], off1[1]])

            # === Per-ROI RANSAC: remove outliers within this ROI ===
            H_roi, inliers_roi = compute_homography(
                kpts0_orig, kpts1_orig, thr=opt.roi_ransac_thr)

            if H_roi is not None:
                # Keep only inliers from this ROI
                inlier_mask = np.array(inliers_roi, dtype=bool)
                kpts0_orig = kpts0_orig[inlier_mask]
                kpts1_orig = kpts1_orig[inlier_mask]
                conf = conf[inlier_mask]

            roi_all_kpts0.append(kpts0_orig)
            roi_all_kpts1.append(kpts1_orig)
            roi_all_conf.append(conf)
            roi_stats.append({
                'roi_idx': roi_idx,
                'raw_matches': raw_count,
                'after_ransac': len(conf),
            })

        # === Merge all ROI inliers ===
        if roi_all_kpts0:
            roi_all_kpts0 = np.vstack(roi_all_kpts0)
            roi_all_kpts1 = np.vstack(roi_all_kpts1)
            roi_all_conf = np.concatenate(roi_all_conf)
        else:
            roi_all_kpts0 = np.empty((0, 2))
            roi_all_kpts1 = np.empty((0, 2))
            roi_all_conf = np.empty(0)

        # === Global confidence sort + limit ===
        if len(roi_all_conf) > 0:
            sorted_idx = np.argsort(-roi_all_conf)[:opt.max_total_matches]
            roi_all_kpts0 = roi_all_kpts0[sorted_idx]
            roi_all_kpts1 = roi_all_kpts1[sorted_idx]
            roi_all_conf = roi_all_conf[sorted_idx]

        total_roi = len(roi_all_kpts0)

        # === Final RANSAC on merged matches ===
        H_roi, inliers_roi = compute_homography(
            roi_all_kpts0, roi_all_kpts1, thr=opt.ransac_thr)
        num_inliers_roi = sum(inliers_roi)

        # Accumulate stats
        stats['total_rois'] += num_rois
        stats['total_original_matches'] += total_original
        stats['total_roi_matches'] += total_roi
        stats['total_original_inliers'] += num_inliers_orig
        stats['total_roi_inliers'] += num_inliers_roi
        if H_orig is not None:
            stats['homo_original_success'] += 1
        if H_roi is not None:
            stats['homo_roi_success'] += 1

        # Store results
        pair_result = {
            'pair': [img_path0.name, img_path1.name],
            'num_common_rois': num_rois,
            'roi_stats': roi_stats,
            'original': {
                'num_matches': total_original,
                'num_inliers': num_inliers_orig,
                'inlier_ratio': num_inliers_orig / total_original if total_original > 0 else 0,
                'homography': H_orig.tolist() if H_orig is not None else None,
            },
            'roi_crop': {
                'num_matches': total_roi,
                'num_inliers': num_inliers_roi,
                'inlier_ratio': num_inliers_roi / total_roi if total_roi > 0 else 0,
                'homography': H_roi.tolist() if H_roi is not None else None,
            },
            'keypoints0': roi_all_kpts0.tolist(),
            'keypoints1': roi_all_kpts1.tolist(),
            'confidence': roi_all_conf.tolist(),
        }
        all_results['pairs'].append(pair_result)

        # Print per-pair result
        print(f'Pair {i}: {img_path0.name[:30]}... ({num_rois} ROIs)')
        if total_original > 0:
            print(f'  Full-image: {total_original} matches, {num_inliers_orig} inliers '
                  f'({num_inliers_orig/total_original*100:.1f}%), H: {"OK" if H_orig is not None else "FAIL"}')
        if total_roi > 0:
            print(f'  ROI-crop:   {total_roi} matches, {num_inliers_roi} inliers '
                  f'({num_inliers_roi/total_roi*100:.1f}%), H: {"OK" if H_roi is not None else "FAIL"}')
        else:
            print(f'  ROI-crop:   0 matches')

    # Save results
    output_file = Path(opt.output_dir) / 'loftr_roi_crop_results.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2)

    # Print summary
    print('\n' + '='*60)
    print('SUMMARY')
    print('='*60)
    n = len(all_results['pairs'])
    if n > 0:
        print(f'Pairs processed:         {n}')
        print(f'Total ROIs matched:      {stats["total_rois"]}')
        print(f'')
        print(f'Full-image avg matches:  {stats["total_original_matches"]/n:.0f}')
        print(f'ROI-crop avg matches:    {stats["total_roi_matches"]/n:.0f}')
        print(f'')
        print(f'Full-image avg inliers:  {stats["total_original_inliers"]/n:.1f}')
        print(f'ROI-crop avg inliers:    {stats["total_roi_inliers"]/n:.1f}')
        print(f'')
        if stats["total_original_matches"] > 0:
            print(f'Full-image inlier ratio: {stats["total_original_inliers"]/stats["total_original_matches"]*100:.1f}%')
        if stats["total_roi_matches"] > 0:
            print(f'ROI-crop inlier ratio:   {stats["total_roi_inliers"]/stats["total_roi_matches"]*100:.1f}%')
        print(f'')
        print(f'Full-image H success:    {stats["homo_original_success"]}/{n} ({stats["homo_original_success"]/n*100:.1f}%)')
        print(f'ROI-crop H success:      {stats["homo_roi_success"]}/{n} ({stats["homo_roi_success"]/n*100:.1f}%)')
    print(f'\nResults saved to {output_file}')


if __name__ == '__main__':
    main()
