"""
SunAngle/metrics_and_vis.py
Planetary Image Registration Evaluation Metrics and Visualization Generator
Phase 3 implementation for SIH26166.
"""

import os
from typing import Dict, Any, Tuple, Optional
import cv2
import numpy as np


def compute_alignment_similarity_metrics(
    img0: np.ndarray,
    img1_warped: np.ndarray
) -> Dict[str, float]:
    """
    Computes registration quality metrics:
    - NCC (Normalized Cross Correlation)
    - Mutual Information (MI)
    - PSNR (Peak Signal to Noise Ratio)
    """
    h, w = img0.shape[:2]
    if img1_warped.shape[:2] != (h, w):
        img1_warped = cv2.resize(img1_warped, (w, h))

    g0 = (img0.astype(np.float32) / 255.0) if img0.max() > 1.0 else img0.astype(np.float32)
    g1 = (img1_warped.astype(np.float32) / 255.0) if img1_warped.max() > 1.0 else img1_warped.astype(np.float32)

    if g0.ndim == 3:
        g0 = cv2.cvtColor(g0, cv2.COLOR_BGR2GRAY)
    if g1.ndim == 3:
        g1 = cv2.cvtColor(g1, cv2.COLOR_BGR2GRAY)

    valid_mask = (g1 > 0.01) & (g0 > 0.01)
    if not np.any(valid_mask):
        return {"ncc": 0.0, "mutual_information": 0.0, "psnr": 0.0}

    v0 = g0[valid_mask]
    v1 = g1[valid_mask]

    # Normalized Cross Correlation
    norm0 = v0 - np.mean(v0)
    norm1 = v1 - np.mean(v1)
    std0 = np.std(v0) + 1e-6
    std1 = np.std(v1) + 1e-6
    ncc = float(np.mean(norm0 * norm1) / (std0 * std1))

    # PSNR
    mse = float(np.mean((v0 - v1) ** 2))
    psnr = float(10.0 * np.log10(1.0 / (mse + 1e-10)))

    # Fast 2D Mutual Information approximation
    hist_2d, _, _ = np.histogram2d(v0, v1, bins=32, range=[[0, 1], [0, 1]])
    pxy = hist_2d / float(np.sum(hist_2d) + 1e-12)
    px = np.sum(pxy, axis=1)
    py = np.sum(pxy, axis=0)
    px_py = px[:, None] * py[None, :]
    nzs = pxy > 0
    mi = float(np.sum(pxy[nzs] * np.log(pxy[nzs] / (px_py[nzs] + 1e-12))))

    return {
        "ncc": round(ncc, 4),
        "mutual_information": round(mi, 4),
        "psnr": round(psnr, 2)
    }


def visualize_matches(
    img0: np.ndarray,
    img1: np.ndarray,
    pts0: np.ndarray,
    pts1: np.ndarray,
    inlier_mask: Optional[np.ndarray] = None,
    out_path: Optional[str] = None,
    max_draw: int = 150
) -> np.ndarray:
    """
    Renders side-by-side match line visualization with green for inliers and red for rejected outliers.
    """
    h0, w0 = img0.shape[:2]
    h1, w1 = img1.shape[:2]

    # Make canvas
    canvas_h = max(h0, h1)
    canvas_w = w0 + w1
    canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)

    def to_bgr(img):
        if img.ndim == 2:
            return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        return img

    canvas[:h0, :w0] = to_bgr(img0)
    canvas[:h1, w0:] = to_bgr(img1)

    if inlier_mask is None:
        inlier_mask = np.ones(len(pts0), dtype=bool)

    total_matches = len(pts0)
    if total_matches == 0:
        if out_path:
            cv2.imwrite(out_path, canvas)
        return canvas

    # Subsample indices for visual clarity
    indices = np.arange(total_matches)
    if total_matches > max_draw:
        # Prioritize inliers
        inlier_indices = indices[inlier_mask]
        outlier_indices = indices[~inlier_mask]
        n_in = min(len(inlier_indices), int(max_draw * 0.85))
        n_out = min(len(outlier_indices), max_draw - n_in)

        selected_in = np.random.choice(inlier_indices, n_in, replace=False) if n_in > 0 else np.array([], dtype=int)
        selected_out = np.random.choice(outlier_indices, n_out, replace=False) if n_out > 0 else np.array([], dtype=int)
        draw_indices = np.concatenate([selected_in, selected_out])
    else:
        draw_indices = indices

    for idx in draw_indices:
        x0, y0 = pts0[idx]
        x1, y1 = pts1[idx]
        is_inlier = bool(inlier_mask[idx])

        color = (0, 230, 70) if is_inlier else (0, 60, 235)  # Green for inlier, Red for outlier
        thickness = 1 if is_inlier else 1
        radius = 3

        pt0 = (int(round(x0)), int(round(y0)))
        pt1 = (int(round(x1 + w0)), int(round(y1)))

        cv2.circle(canvas, pt0, radius, color, -1)
        cv2.circle(canvas, pt1, radius, color, -1)
        cv2.line(canvas, pt0, pt1, color, thickness, cv2.LINE_AA)

    # Header label
    inlier_count = int(np.sum(inlier_mask))
    header_text = f"Matches: {total_matches} | Inliers: {inlier_count} ({inlier_count/total_matches*100:.1f}%)"
    cv2.putText(canvas, header_text, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

    if out_path:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        cv2.imwrite(out_path, canvas)

    return canvas


def visualize_spatial_distribution(
    img_shape: Tuple[int, int],
    pts: np.ndarray,
    out_path: Optional[str] = None,
    grid_size: int = 8
) -> np.ndarray:
    """
    Renders 2D spatial density grid map displaying coverage and point concentration across terrain.
    """
    h, w = img_shape[:2]
    density_map = np.zeros((h, w), dtype=np.float32)

    gh = h / float(grid_size)
    gw = w / float(grid_size)

    counts = np.zeros((grid_size, grid_size), dtype=np.int32)
    for x, y in pts:
        gx = min(max(int(x / gw), 0), grid_size - 1)
        gy = min(max(int(y / gh), 0), grid_size - 1)
        counts[gy, gx] += 1

    max_c = max(1, counts.max())
    for gy in range(grid_size):
        for gx in range(grid_size):
            y0, y1 = int(gy * gh), int((gy + 1) * gh)
            x0, x1 = int(gx * gw), int((gx + 1) * gw)
            density_map[y0:y1, x0:x1] = counts[gy, gx] / float(max_c)

    density_u8 = (density_map * 255.0).astype(np.uint8)
    color_density = cv2.applyColorMap(density_u8, cv2.COLORMAP_VIRIDIS)

    # Draw grid lines
    for gy in range(grid_size + 1):
        y = int(gy * gh)
        cv2.line(color_density, (0, y), (w, y), (200, 200, 200), 1)
    for gx in range(grid_size + 1):
        x = int(gx * gw)
        cv2.line(color_density, (x, 0), (x, h), (200, 200, 200), 1)

    # Plot point dots
    for x, y in pts:
        cv2.circle(color_density, (int(x), int(y)), 2, (255, 255, 255), -1)

    if out_path:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        cv2.imwrite(out_path, color_density)

    return color_density
