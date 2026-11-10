"""
LoFTR inference script for AutoX sequential images.
Outputs: keypoints, matches, confidence scores, and homography for each consecutive pair.

Usage:
    # 方式1：使用 kornia（推荐，无需下载权重）
    python scripts/run_loftr_autox.py --use_kornia

    # 方式2：使用本地权重
    python scripts/run_loftr_autox.py --weights weights/outdoor_ds.ckpt

    # 可选参数
    python scripts/run_loftr_autox.py --use_kornia \
        --conf_thr 0.2 \
        --top_k 5000 \
        --resize 1280 960

    # 只测试5对图像
    python scripts/run_loftr_autox.py --weights weights/outdoor_ds.ckpt --num_pairs 5

Output:
    - results saved to outputs/autox_loftr/loftr_results.json
    - contains: keypoints0, keypoints1, confidence, homography, num_inliers for each pair
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


def compute_homography(kpts0, kpts1, method=cv2.RANSAC, thr=5.0):
    """Compute homography matrix from matched keypoints."""
    if len(kpts0) < 4:
        return None, []
    
    kpts0_np = np.float32(kpts0)
    kpts1_np = np.float32(kpts1)
    
    H, mask = cv2.findHomography(kpts0_np, kpts1_np, method, thr)
    if mask is not None:
        inliers = mask.ravel().tolist()
    else:
        inliers = []
    
    return H, inliers


def main():
    parser = argparse.ArgumentParser(description='LoFTR inference on AutoX images')
    parser.add_argument('--images_dir', type=str, 
                        default=r'E:\毕业材料\大论文\数据集\AutoX\images',
                        help='Path to images directory')
    parser.add_argument('--weights', type=str, 
                        default=r'E:\workspacePython\LoFTR\weights\outdoor_ds.ckpt',
                        help='Path to LoFTR checkpoint')
    parser.add_argument('--output_dir', type=str, 
                        default=r'E:\workspacePython\LoFTR\outputs\autox_loftr',
                        help='Output directory for results')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda or cpu)')
    parser.add_argument('--resize', type=int, nargs='+', default=[1024],
                        help='Resize max dimension of input images. Default 640.')
    parser.add_argument('--top_k', type=int, default=5000,
                        help='Maximum number of matches to keep')
    parser.add_argument('--conf_thr', type=float, default=0.2,
                        help='Confidence threshold for matches')
    parser.add_argument('--use_kornia', action='store_true',
                        help='Use kornia LoFTR instead of local weights')
    parser.add_argument('--num_pairs', type=int, default=0,
                        help='Number of pairs to process (0=all)')
    
    opt = parser.parse_args()
    
    # Create output directory
    Path(opt.output_dir).mkdir(parents=True, exist_ok=True)
    
    # Auto-detect device
    if opt.device == 'cuda' and not torch.cuda.is_available():
        print('CUDA not available, using CPU instead')
        opt.device = 'cpu'
    
    # Initialize LoFTR
    print('Loading LoFTR model...')
    if opt.use_kornia:
        from kornia.feature import LoFTR as KorniaLoFTR
        matcher = KorniaLoFTR(pretrained='outdoor')
        print('Using kornia LoFTR (outdoor pretrained)')
    else:
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from src.loftr import LoFTR, default_cfg
        matcher = LoFTR(config=default_cfg)
        matcher.load_state_dict(torch.load(opt.weights, map_location='cpu', weights_only=True)['state_dict'])
        print(f'Using local weights: {opt.weights}')
    
    matcher = matcher.eval().to(device=opt.device)
    print('Model loaded successfully!')
    
    # Get all images sorted
    image_dir = Path(opt.images_dir)
    # Use set to avoid duplicates on case-insensitive filesystems (Windows)
    image_files = sorted(set(
        list(image_dir.glob('*.jpg')) + list(image_dir.glob('*.png')) + list(image_dir.glob('*.JPG'))
        + list(image_dir.glob('*.jpeg')) + list(image_dir.glob('*.JPEG'))
    ))
    print(f'Found {len(image_files)} images')
    
    # Limit number of pairs if specified
    num_pairs = len(image_files) - 1
    if opt.num_pairs > 0:
        num_pairs = min(opt.num_pairs, len(image_files) - 1)
        print(f'Processing first {num_pairs} pairs')
    
    # Store all results
    all_results = {
        'images': [str(f.name) for f in image_files],
        'num_images': len(image_files),
        'matches': []
    }
    
    # Process consecutive pairs
    for i in tqdm(range(num_pairs), desc='Processing pairs'):
        img_path0 = image_files[i]
        img_path1 = image_files[i + 1]
        
        # Read images (use np.fromfile to support Chinese paths)
        img0 = cv2.imdecode(np.fromfile(str(img_path0), dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
        img1 = cv2.imdecode(np.fromfile(str(img_path1), dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
        
        if img0 is None or img1 is None:
            print(f'Warning: Failed to read images {img_path0} or {img_path1}')
            continue
        
        # Resize if needed (ensure dimensions are multiples of 32 for LoFTR FPN)
        if opt.resize[0] != -1:
            max_dim = max(img0.shape)
            scale = opt.resize[0] / max_dim
            new_w = int(img0.shape[1] * scale)
            new_h = int(img0.shape[0] * scale)
            # Round up to nearest multiple of 32
            new_w = (new_w + 31) // 32 * 32
            new_h = (new_h + 31) // 32 * 32
            img0 = cv2.resize(img0, (new_w, new_h))
            img1 = cv2.resize(img1, (new_w, new_h))
        
        # Prepare batch
        if opt.use_kornia:
            # kornia expects (B, 1, H, W)
            img0_tensor = torch.from_numpy(img0).float().to(opt.device) / 255.
            img1_tensor = torch.from_numpy(img1).float().to(opt.device) / 255.
            batch = {
                'image0': img0_tensor.unsqueeze(0).unsqueeze(0),
                'image1': img1_tensor.unsqueeze(0).unsqueeze(0),
            }
        else:
            # Original LoFTR expects (B, 1, H, W) as well
            batch = {
                'image0': torch.from_numpy(img0)[None][None].float().to(opt.device) / 255.,
                'image1': torch.from_numpy(img1)[None][None].float().to(opt.device) / 255.,
            }
        
        # Run LoFTR
        with torch.no_grad():
            matcher(batch)
        
        # Get results
        mkpts0 = batch['mkpts0_f'].cpu().numpy()
        mkpts1 = batch['mkpts1_f'].cpu().numpy()
        mconf = batch['mconf'].cpu().numpy()
        
        # Filter by confidence
        conf_mask = mconf >= opt.conf_thr
        mkpts0 = mkpts0[conf_mask]
        mkpts1 = mkpts1[conf_mask]
        mconf = mconf[conf_mask]
        
        # Sort by confidence and keep top_k
        if len(mconf) > 0:
            sorted_idx = np.argsort(-mconf)[:opt.top_k]
            mkpts0 = mkpts0[sorted_idx]
            mkpts1 = mkpts1[sorted_idx]
            mconf = mconf[sorted_idx]
        
        # Compute homography
        H, inliers = compute_homography(mkpts0, mkpts1)
        
        # Store results for this pair
        pair_result = {
            'pair': [str(img_path0.name), str(img_path1.name)],
            'num_matches': len(mkpts0),
            'keypoints0': mkpts0.tolist(),
            'keypoints1': mkpts1.tolist(),
            'confidence': mconf.tolist(),
            'homography': H.tolist() if H is not None else None,
            'num_inliers': sum(inliers) if inliers else 0,
            'inlier_mask': inliers
        }
        all_results['matches'].append(pair_result)
        
        print(f'Pair {i}: {img_path0.name} <-> {img_path1.name}, '
              f'Matches: {len(mkpts0)}, Inliers: {sum(inliers) if inliers else 0}, '
              f'H: {"OK" if H is not None else "FAIL"}')
    
    # Save results
    output_file = Path(opt.output_dir) / 'loftr_results.json'
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f'\nResults saved to {output_file}')
    
    # Print summary
    n = len(all_results['matches'])
    total_matches = sum(r['num_matches'] for r in all_results['matches'])
    avg_matches = total_matches / n if n > 0 else 0
    total_inliers = sum(r['num_inliers'] for r in all_results['matches'])
    avg_inliers = total_inliers / n if n > 0 else 0
    homo_success = sum(1 for r in all_results['matches'] if r['homography'] is not None)
    print(f'\nSummary:')
    print(f'  Total pairs processed:    {n}')
    print(f'  Total matches:            {total_matches}')
    print(f'  Average matches per pair: {avg_matches:.1f}')
    print(f'  Total inliers:            {total_inliers}')
    print(f'  Average inliers per pair: {avg_inliers:.1f}')
    print(f'  Homography success rate:  {homo_success}/{n} ({homo_success/n*100:.1f}%)' if n > 0 else
          f'  Homography success rate:  0/0')


if __name__ == '__main__':
    main()
