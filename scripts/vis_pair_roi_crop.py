"""
CVPR-style color LoFTR matching visualization for a specific image pair
using ROI-crop results.

Usage:
    python3 scripts/vis_pair_roi_crop.py
"""

import json
import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from matplotlib.patches import Circle
from pathlib import Path


# ── Configuration ──
JSON_PATH = 'outputs/autox_loftr_roi_crop/loftr_roi_crop_results.json'
IMAGES_DIR = r'/mnt/e/毕业材料/大论文/数据集/AutoX/images'
OUTPUT_DIR = 'outputs/autox_loftr_roi_crop/vis_paper'

IMG0_NAME = '24-05-18-20240518-2#14-rep0-DSC-RX0-1715996905111.JPG'
IMG1_NAME = '24-05-18-20240518-2#16-rep0-DSC-RX0-1715996927029.JPG'

MAX_MATCHES = 200
DPI = 200
RANSAC_THR = 4.0


def load_image_color(path):
    img = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Cannot read: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def find_pair(pairs_data, name0, name1):
    for pair in pairs_data:
        p = pair['pair']
        if (p[0] == name0 and p[1] == name1) or (p[0] == name1 and p[1] == name0):
            return pair
    return None


def compute_inlier_mask(kpts0, kpts1, H, thr=RANSAC_THR):
    pts0_h = np.hstack([kpts0, np.ones((len(kpts0), 1))])
    pts1_proj = (H @ pts0_h.T).T
    pts1_proj = pts1_proj[:, :2] / pts1_proj[:, 2:3]
    errors = np.linalg.norm(pts1_proj - kpts1, axis=1)
    return errors < thr


def random_colors(n, seed=42):
    """Generate n visually distinct colors using golden ratio hue spacing."""
    rng = np.random.RandomState(seed)
    hues = (np.arange(n) * 0.618033988749895) % 1.0
    rng.shuffle(hues)
    colors = np.zeros((n, 4))
    for i, h in enumerate(hues):
        r, g, b = matplotlib.colors.hsv_to_rgb([h, 0.85, 0.95])
        colors[i] = [r, g, b, 0.85]
    return colors


def turbo_colormap(conf, alpha=0.8):
    """Map confidence to turbo colormap (warm=high, cool=low)."""
    norm = (conf - conf.min()) / (conf.max() - conf.min() + 1e-8)
    cmap = plt.cm.turbo
    colors = cmap(norm)
    colors[:, 3] = alpha
    return colors


def cvpr_matching_figure(img0, img1, mkpts0, mkpts1, colors,
                         kpt_edge_color='white', kpt_size=8,
                         line_width=0.6, text_lines=None, dpi=200):
    """CVPR-style: color images, thin match lines, white-edged keypoints, no axes."""
    h0, w0 = img0.shape[:2]
    h1, w1 = img1.shape[:2]

    total_w = w0 + w1
    fig_w = 16
    fig_h = fig_w * max(h0, h1) / total_w
    fig_h = max(fig_h, 5)

    fig, (ax0, ax1) = plt.subplots(
        1, 2, figsize=(fig_w, fig_h), dpi=dpi,
        gridspec_kw={'width_ratios': [w0, w1], 'wspace': 0.02}
    )

    ax0.imshow(img0)
    ax1.imshow(img1)

    for ax in (ax0, ax1):
        ax.set_axis_off()

    if len(mkpts0) > 0:
        fig.canvas.draw()
        transFig = fig.transFigure.inverted()
        fk0 = transFig.transform(ax0.transData.transform(mkpts0))
        fk1 = transFig.transform(ax1.transData.transform(mkpts1))

        # Draw match lines
        for i in range(len(mkpts0)):
            c = colors[i]
            alpha = c[3] if len(c) > 3 else 0.8
            fig.add_artist(mlines.Line2D(
                (fk0[i, 0], fk1[i, 0]), (fk0[i, 1], fk1[i, 1]),
                transform=fig.transFigure,
                color=c[:3], linewidth=line_width, alpha=alpha,
                zorder=1
            ))

        # White-edged keypoints on each image
        for ax, mkpts in [(ax0, mkpts0), (ax1, mkpts1)]:
            ax.scatter(
                mkpts[:, 0], mkpts[:, 1],
                c=colors[:, :3], s=kpt_size,
                edgecolors=kpt_edge_color, linewidths=0.8,
                zorder=10
            )

    if text_lines:
        fig.text(0.005, 0.98, '\n'.join(text_lines),
                 transform=ax0.transAxes, fontsize=11, va='top', ha='left',
                 color='white', weight='bold',
                 bbox=dict(boxstyle='round,pad=0.3', facecolor='black',
                           alpha=0.5, edgecolor='none'))

    return fig


def main():
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    print(f"Loading JSON: {JSON_PATH}")
    with open(JSON_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)

    pairs_data = data.get('pairs', data.get('matches', []))
    pair = find_pair(pairs_data, IMG0_NAME, IMG1_NAME)
    if pair is None:
        print("ERROR: Pair not found in JSON!")
        return

    print(f"Found pair: {pair['pair']}")

    kpts0 = np.array(pair['keypoints0'])
    kpts1 = np.array(pair['keypoints1'])
    conf = np.array(pair['confidence'])
    n_total = len(kpts0)
    print(f"Total matches: {n_total}")

    roi_crop = pair.get('roi_crop', {})
    H = np.array(roi_crop['homography']) if roi_crop.get('homography') else None
    inlier_mask = compute_inlier_mask(kpts0, kpts1, H, RANSAC_THR) if H is not None else None
    n_inliers = int(inlier_mask.sum()) if inlier_mask is not None else 0
    print(f"Inliers (roi_crop): {n_inliers}")

    img0 = load_image_color(Path(IMAGES_DIR) / IMG0_NAME)
    img1 = load_image_color(Path(IMAGES_DIR) / IMG1_NAME)
    print(f"Image sizes: {img0.shape}, {img1.shape}")

    # ── Figure 1: Confidence-colored matches (turbo colormap) ──
    n_show = min(n_total, MAX_MATCHES)
    idx = np.argsort(-conf)[:n_show]
    mkpts0 = kpts0[idx]
    mkpts1 = kpts1[idx]
    colors_turbo = turbo_colormap(conf[idx], alpha=0.7)

    fig1 = cvpr_matching_figure(
        img0, img1, mkpts0, mkpts1, colors_turbo,
        text_lines=[f'LoFTR (ROI-Crop) — Top {n_show} matches (confidence)']
    )
    out1 = Path(OUTPUT_DIR) / 'cvpr_confidence.png'
    fig1.savefig(str(out1), bbox_inches='tight', pad_inches=0.05, dpi=DPI)
    plt.close(fig1)
    print(f"Saved: {out1}")

    # ── Figure 2: Random distinct colors per match ──
    colors_rand = random_colors(n_show, seed=2024)
    colors_rand[:, 3] = 0.75

    fig2 = cvpr_matching_figure(
        img0, img1, mkpts0, mkpts1, colors_rand,
        text_lines=[f'LoFTR (ROI-Crop) — Top {n_show} matches']
    )
    out2 = Path(OUTPUT_DIR) / 'cvpr_random_colors.png'
    fig2.savefig(str(out2), bbox_inches='tight', pad_inches=0.05, dpi=DPI)
    plt.close(fig2)
    print(f"Saved: {out2}")

    # ── Figure 3: Inliers (green) vs outliers (light gray) ──
    if inlier_mask is not None:
        inlier_idx = np.where(inlier_mask[idx])[0]
        outlier_idx = np.where(~inlier_mask[idx])[0]

        all_idx = np.concatenate([outlier_idx, inlier_idx])
        mkpts0_i = mkpts0[all_idx]
        mkpts1_i = mkpts1[all_idx]

        colors_i = np.zeros((len(all_idx), 4))
        n_out = len(outlier_idx)
        colors_i[:n_out] = [0.7, 0.7, 0.7, 0.15]
        colors_i[n_out:] = [0.0, 0.85, 0.3, 0.95]

        fig3 = cvpr_matching_figure(
            img0, img1, mkpts0_i, mkpts1_i, colors_i,
            line_width=0.8,
            text_lines=[
                f'LoFTR (ROI-Crop)',
                f'{n_total} matches → {n_inliers} inliers ({n_inliers/n_total*100:.1f}%)'
            ]
        )
        out3 = Path(OUTPUT_DIR) / 'cvpr_inliers.png'
        fig3.savefig(str(out3), bbox_inches='tight', pad_inches=0.05, dpi=DPI)
        plt.close(fig3)
        print(f"Saved: {out3}")

    # ── Figure 4: Inliers only with random colors ──
    if inlier_mask is not None and n_inliers > 0:
        inlier_global = np.where(inlier_mask)[0]
        inlier_sorted = inlier_global[np.argsort(-conf[inlier_global])]
        mkpts0_inl = kpts0[inlier_sorted]
        mkpts1_inl = kpts1[inlier_sorted]

        colors_inl = random_colors(n_inliers, seed=7)
        colors_inl[:, 3] = 0.9

        fig4 = cvpr_matching_figure(
            img0, img1, mkpts0_inl, mkpts1_inl, colors_inl,
            kpt_size=12, line_width=1.0,
            text_lines=[f'LoFTR (ROI-Crop) — {n_inliers} inlier matches']
        )
        out4 = Path(OUTPUT_DIR) / 'cvpr_inliers_only.png'
        fig4.savefig(str(out4), bbox_inches='tight', pad_inches=0.05, dpi=DPI)
        plt.close(fig4)
        print(f"Saved: {out4}")

    # ── Figure 5: Confidence-colored inliers only ──
    if inlier_mask is not None and n_inliers > 0:
        inlier_global = np.where(inlier_mask)[0]
        inlier_sorted = inlier_global[np.argsort(-conf[inlier_global])]
        mkpts0_inl = kpts0[inlier_sorted]
        mkpts1_inl = kpts1[inlier_sorted]

        colors_inl_conf = turbo_colormap(conf[inlier_sorted], alpha=0.9)

        fig5 = cvpr_matching_figure(
            img0, img1, mkpts0_inl, mkpts1_inl, colors_inl_conf,
            kpt_size=12, line_width=1.0,
            text_lines=[f'LoFTR (ROI-Crop) — {n_inliers} inliers (confidence)']
        )
        out5 = Path(OUTPUT_DIR) / 'cvpr_inliers_confidence.png'
        fig5.savefig(str(out5), bbox_inches='tight', pad_inches=0.05, dpi=DPI)
        plt.close(fig5)
        print(f"Saved: {out5}")

    print(f"\nDone! All CVPR-style figures saved to {OUTPUT_DIR}")


if __name__ == '__main__':
    main()
