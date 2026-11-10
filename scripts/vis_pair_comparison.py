"""
CVPR-style comparison figure: Original LoFTR (1024x1024) vs ROI-Crop (1024x1024).

Top row: full-image matches (outliers faint gray, inliers bright turbo colors)
Bottom row: ROI-crop matches (same style)

Usage:
    python3 scripts/vis_pair_comparison.py
"""

import json
import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from pathlib import Path


# ── Configuration ──
FULL_IMAGE_JSON = 'outputs/autox_loftr/loftr_results.json'
ROI_CROP_JSON = 'outputs/autox_loftr_roi_crop/loftr_roi_crop_results.json'
IMAGES_DIR = r'/mnt/e/毕业材料/大论文/数据集/AutoX/images'
OUTPUT_DIR = 'outputs/autox_loftr_roi_crop/vis_paper'

IMG0_NAME = '24-05-18-20240518-2#14-rep0-DSC-RX0-1715996905111.JPG'
IMG1_NAME = '24-05-18-20240518-2#16-rep0-DSC-RX0-1715996927029.JPG'

MAX_DISPLAY = 50
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


def turbo_colormap(conf, alpha=0.85):
    norm = (conf - conf.min()) / (conf.max() - conf.min() + 1e-8)
    cmap = plt.cm.turbo
    colors = cmap(norm)
    colors[:, 3] = alpha
    return colors


def draw_matching_row(fig, axes, mkpts0, mkpts1, colors, line_width=1.2, kpt_size=10):
    ax0, ax1 = axes
    if len(mkpts0) == 0:
        return

    fig.canvas.draw()
    transFig = fig.transFigure.inverted()
    fk0 = transFig.transform(ax0.transData.transform(mkpts0))
    fk1 = transFig.transform(ax1.transData.transform(mkpts1))

    for i in range(len(mkpts0)):
        c = colors[i]
        alpha = c[3] if len(c) > 3 else 0.75
        fig.add_artist(mlines.Line2D(
            (fk0[i, 0], fk1[i, 0]), (fk0[i, 1], fk1[i, 1]),
            transform=fig.transFigure,
            color=c[:3], linewidth=line_width, alpha=alpha,
            zorder=1
        ))

    for ax, mkpts in [(ax0, mkpts0), (ax1, mkpts1)]:
        ax.scatter(
            mkpts[:, 0], mkpts[:, 1],
            c=colors[:, :3], s=kpt_size,
            edgecolors='white', linewidths=0.8,
            zorder=10
        )


def prepare_display_data(kpts0, kpts1, conf, max_display):
    """Top N matches sorted by confidence, colored by turbo colormap."""
    n_show = min(len(kpts0), max_display)
    idx = np.argsort(-conf)[:n_show]
    mkpts0 = kpts0[idx]
    mkpts1 = kpts1[idx]
    c = conf[idx]
    colors = turbo_colormap(c, alpha=0.75)
    return mkpts0, mkpts1, colors


def main():
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    # ── Load images ──
    img0_color = load_image_color(Path(IMAGES_DIR) / IMG0_NAME)
    img1_color = load_image_color(Path(IMAGES_DIR) / IMG1_NAME)
    print(f"Image sizes: {img0_color.shape}, {img1_color.shape}")

    # ── Load full-image LoFTR results ──
    with open(FULL_IMAGE_JSON, 'r', encoding='utf-8') as f:
        full_data = json.load(f)
    full_pair = find_pair(full_data.get('matches', []), IMG0_NAME, IMG1_NAME)
    if full_pair is None:
        print("ERROR: Pair not found in full-image JSON!")
        return

    fk0 = np.array(full_pair['keypoints0'])
    fk1 = np.array(full_pair['keypoints1'])
    fc = np.array(full_pair['confidence'])
    fH = np.array(full_pair['homography']) if full_pair.get('homography') else None

    resize = 1024
    scale0 = resize / max(img0_color.shape[:2])
    scale1 = resize / max(img1_color.shape[:2])
    fk0_orig = fk0.copy(); fk0_orig[:, 0] /= scale0; fk0_orig[:, 1] /= scale0
    fk1_orig = fk1.copy(); fk1_orig[:, 0] /= scale1; fk1_orig[:, 1] /= scale1

    full_inlier = compute_inlier_mask(fk0, fk1, fH, RANSAC_THR) if fH is not None else None
    full_n_inliers = int(full_inlier.sum()) if full_inlier is not None else 0
    print(f"Full-image: {len(fk0)} matches, {full_n_inliers} inliers")

    # ── Load ROI-crop results ──
    with open(ROI_CROP_JSON, 'r', encoding='utf-8') as f:
        roi_data = json.load(f)
    roi_pair = find_pair(roi_data.get('pairs', []), IMG0_NAME, IMG1_NAME)
    if roi_pair is None:
        print("ERROR: Pair not found in ROI-crop JSON!")
        return

    rk0 = np.array(roi_pair['keypoints0'])
    rk1 = np.array(roi_pair['keypoints1'])
    rc = np.array(roi_pair['confidence'])
    rH = np.array(roi_pair['roi_crop']['homography']) if roi_pair['roi_crop'].get('homography') else None

    roi_inlier = compute_inlier_mask(rk0, rk1, rH, RANSAC_THR) if rH is not None else None
    roi_n_inliers = int(roi_inlier.sum()) if roi_inlier is not None else 0
    print(f"ROI-Crop:   {len(rk0)} matches, {roi_n_inliers} inliers")

    # ── Prepare display data ──
    full_mkpts0, full_mkpts1, full_colors = prepare_display_data(
        fk0_orig, fk1_orig, fc, MAX_DISPLAY)
    roi_mkpts0, roi_mkpts1, roi_colors = prepare_display_data(
        rk0, rk1, rc, MAX_DISPLAY)

    # ══════════════════════════════════════════════════════════
    # Figure: 2-row comparison
    # ══════════════════════════════════════════════════════════
    h0, w0 = img0_color.shape[:2]
    h1, w1 = img1_color.shape[:2]
    total_w = w0 + w1

    fig_w = 18
    row_h = fig_w * max(h0, h1) / total_w
    fig_h = row_h * 2 + 0.5
    fig_h = max(fig_h, 10)

    fig, axes = plt.subplots(
        2, 2, figsize=(fig_w, fig_h), dpi=DPI,
        gridspec_kw={
            'width_ratios': [w0, w1],
            'height_ratios': [1, 1],
            'wspace': 0.02,
            'hspace': 0.02,
        }
    )

    for row in range(2):
        axes[row, 0].imshow(img0_color)
        axes[row, 1].imshow(img1_color)
        for ax in axes[row]:
            ax.set_axis_off()

    draw_matching_row(fig, axes[0], full_mkpts0, full_mkpts1, full_colors, line_width=1.2)
    draw_matching_row(fig, axes[1], roi_mkpts0, roi_mkpts1, roi_colors, line_width=1.2)

    # ── Text labels ──
    label_style = dict(fontsize=11, va='top', ha='left',
                       color='white', weight='bold',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='black',
                                 alpha=0.5, edgecolor='none'))

    n_full_show = min(len(fk0), MAX_DISPLAY)
    n_roi_show = min(len(rk0), MAX_DISPLAY)

    axes[0, 0].text(0.02, 0.98,
                    f'LoFTR (1024\u00d71024) \u2014 Top {min(len(fk0), MAX_DISPLAY)} matches ({full_n_inliers} inliers)',
                    transform=axes[0, 0].transAxes, **label_style)

    axes[1, 0].text(0.02, 0.98,
                    f'LoFTR (ROI-Crop) \u2014 Top {min(len(rk0), MAX_DISPLAY)} matches ({roi_n_inliers} inliers)',
                    transform=axes[1, 0].transAxes, **label_style)

    out_path = Path(OUTPUT_DIR) / 'cvpr_comparison_full_vs_roi.png'
    fig.savefig(str(out_path), bbox_inches='tight', pad_inches=0.05, dpi=DPI)
    plt.close(fig)
    print(f"Saved: {out_path}")


if __name__ == '__main__':
    main()
