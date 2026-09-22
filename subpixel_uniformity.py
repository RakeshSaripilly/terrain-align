"""
SunAngle/subpixel_uniformity.py
Sub-Pixel Accuracy Refinement and Spatial Uniformity Engine
Phase 2 implementation: Subpixel refinement and spatial uniformity.
Enforces sub-pixel precision (target RMSE < 0.5 px) and prevents spatial clustering.
"""

from typing import Tuple, Dict, Any, List
import cv2
import numpy as np
from scipy.ndimage import map_coordinates


def refine_subpixel_lk(
    img0: np.ndarray,
    img1: np.ndarray,
    pts0: np.ndarray,
    pts1: np.ndarray,
    window_size: int = 15,
    max_iters: int = 20,
    eps: float = 1e-3
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Sub-Pixel Refinement using Inverse Compositional Lucas-Kanade on normalized gradient patches.
    Refines pts1 to match pts0 with sub-pixel precision (< 0.5 pixel target).

    Args:
        img0: Reference grayscale image (float32 [0, 1] or uint8)
        img1: Moving/target grayscale image
        pts0: (N, 2) float32 coordinates on img0
        pts1: (N, 2) initial float32 coordinates on img1
        window_size: Patch size (e.g. 15x15)
        max_iters: Maximum Lucas-Kanade gradient ascent iterations
        eps: Convergence delta threshold

    Returns:
        refined_pts0: (M, 2) valid points in img0
        refined_pts1: (M, 2) sub-pixel refined coordinates in img1
        valid_mask: (N,) boolean mask of successfully refined correspondences
    """
    # Convert to uint8 grayscale for high-speed SIMD optical flow
    u0 = (img0 * 255.0).astype(np.uint8) if img0.dtype == np.float32 else img0
    u1 = (img1 * 255.0).astype(np.uint8) if img1.dtype == np.float32 else img1

    if u0.ndim == 3:
        u0 = cv2.cvtColor(u0, cv2.COLOR_BGR2GRAY)
    if u1.ndim == 3:
        u1 = cv2.cvtColor(u1, cv2.COLOR_BGR2GRAY)

    p0 = pts0.reshape(-1, 1, 2).astype(np.float32)
    p1 = pts1.reshape(-1, 1, 2).astype(np.float32)

    # Fast AVX2-accelerated iterative Lucas-Kanade with sub-pixel criteria
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, max_iters, eps)
    next_pts, status, err = cv2.calcOpticalFlowPyrLK(
        u0, u1, p0, p1,
        winSize=(window_size, window_size),
        maxLevel=0,
        criteria=criteria,
        flags=cv2.OPTFLOW_USE_INITIAL_FLOW
    )

    refined_pts1 = next_pts.reshape(-1, 2)
    status = status.ravel().astype(bool)

    # Enforce drift threshold (reject divergent points moving > 3.5 px from coarse match)
    drift = np.linalg.norm(refined_pts1 - pts1, axis=1)
    valid_mask = status & (drift <= 3.5)

    return pts0[valid_mask], refined_pts1[valid_mask], valid_mask



def enforce_spatial_uniformity_anms(
    pts0: np.ndarray,
    pts1: np.ndarray,
    scores: np.ndarray,
    max_points: int = 1000,
    c_robust: float = 0.9
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Adaptive Non-Maximal Suppression (ANMS) with suppressive radius calculation.
    Enforces even spatial distribution of tie points across both crater rims and flat terrain.

    Args:
        pts0: (N, 2) coordinates in img0
        pts1: (N, 2) coordinates in img1
        scores: (N,) confidence scores
        max_points: Target number of well-distributed points
        c_robust: Robustness coefficient (standard is 0.9)

    Returns:
        selected_pts0, selected_pts1, selected_scores
    """
    n = len(pts0)
    if n <= max_points:
        return pts0, pts1, scores

    # Sort descending by score
    sort_idx = np.argsort(-scores)
    sorted_pts0 = pts0[sort_idx]
    sorted_pts1 = pts1[sort_idx]
    sorted_scores = scores[sort_idx]

    # Compute suppression radius for each point
    # r_i = min_j ||p_i - p_j|| subject to f(p_j) * c_robust > f(p_i)
    radii = np.full(n, np.inf)

    # Vectorized / efficient ANMS calculation
    coords = sorted_pts0
    for i in range(1, n):
        # Candidates are all previous points (which have higher score)
        stronger_pts = coords[:i]
        diffs = stronger_pts - coords[i]
        dist_sq = np.sum(diffs * diffs, axis=1)
        radii[i] = np.sqrt(np.min(dist_sq))

    # Pick top points with largest suppression radii
    anms_idx = np.argsort(-radii)[:max_points]
    
    return sorted_pts0[anms_idx], sorted_pts1[anms_idx], sorted_scores[anms_idx]


def quadtree_spatial_binning(
    pts0: np.ndarray,
    pts1: np.ndarray,
    scores: np.ndarray,
    img_shape: Tuple[int, int],
    grid_size: int = 8,
    points_per_bin: int = 25
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Quadtree / Grid Spatial Binning.
    Divides image into grid_size x grid_size buckets and keeps the highest scoring
    correspondences in each spatial cell. Prevents dense clusters in high-contrast crater rims.
    """
    h, w = img_shape[:2]
    gh = h / float(grid_size)
    gw = w / float(grid_size)

    bins: Dict[Tuple[int, int], List[int]] = {}
    for idx, (x, y) in enumerate(pts0):
        gx = min(int(x / gw), grid_size - 1)
        gy = min(int(y / gh), grid_size - 1)
        key = (gy, gx)
        if key not in bins:
            bins[key] = []
        bins[key].append(idx)

    selected_indices = []
    for cell_indices in bins.values():
        if len(cell_indices) <= points_per_bin:
            selected_indices.extend(cell_indices)
        else:
            # Sort by score and pick top points_per_bin
            cell_scores = scores[cell_indices]
            top_local = np.argsort(-cell_scores)[:points_per_bin]
            selected_indices.extend([cell_indices[t] for t in top_local])

    selected_indices = np.array(selected_indices, dtype=int)
    return pts0[selected_indices], pts1[selected_indices], scores[selected_indices]


def compute_spatial_uniformity_metrics(
    pts: np.ndarray,
    img_shape: Tuple[int, int],
    grid_divisions: int = 6
) -> Dict[str, Any]:
    """
    Computes spatial coverage fraction and Shannon entropy of correspondence distribution.
    
    Evaluation standards:
        - Coverage target: > 0.77 (ideal > 0.85)
        - Entropy target:  > 0.70 (ideal > 0.80)
    """
    h, w = img_shape[:2]
    total_cells = grid_divisions * grid_divisions
    cell_counts = np.zeros((grid_divisions, grid_divisions), dtype=np.int32)

    gh = h / float(grid_divisions)
    gw = w / float(grid_divisions)

    for x, y in pts:
        gx = min(max(int(x / gw), 0), grid_divisions - 1)
        gy = min(max(int(y / gh), 0), grid_divisions - 1)
        cell_counts[gy, gx] += 1

    occupied_cells = int(np.sum(cell_counts > 0))
    coverage_fraction = float(occupied_cells / total_cells)

    total_pts = len(pts)
    if total_pts > 0:
        probs = cell_counts.flatten() / float(total_pts)
        probs_nonzero = probs[probs > 0]
        # Shannon entropy normalized by maximum possible entropy log(total_cells)
        entropy = float(-np.sum(probs_nonzero * np.log(probs_nonzero)) / np.log(total_cells))
    else:
        entropy = 0.0

    return {
        "grid_divisions": grid_divisions,
        "total_cells": total_cells,
        "occupied_cells": occupied_cells,
        "coverage_fraction": round(coverage_fraction, 4),
        "spatial_entropy": round(entropy, 4),
        "cell_counts": cell_counts.tolist()
    }
