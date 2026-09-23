"""Classical baselines and evaluation helpers for the SunAngle pipeline."""

from typing import Any, Dict

import cv2
import numpy as np


def compute_metrics(
    mkpts0: np.ndarray,
    mkpts1: np.ndarray,
    homography_gt: np.ndarray | None = None,
) -> Dict[str, Any]:
    """Return basic correspondence counts and optional ground-truth error."""
    metrics: Dict[str, Any] = {"num_matches": int(len(mkpts0))}
    if homography_gt is not None and len(mkpts0) > 0:
        projected = cv2.perspectiveTransform(
            np.asarray(mkpts0, dtype=np.float32).reshape(-1, 1, 2),
            np.asarray(homography_gt, dtype=np.float64),
        ).reshape(-1, 2)
        errors = np.linalg.norm(projected - np.asarray(mkpts1), axis=1)
        metrics["ground_truth_rmse"] = float(np.sqrt(np.mean(errors ** 2)))
    return metrics


def sift_baseline(img0_path: str, img1_path: str) -> int:
    """Count Lowe-ratio-filtered SIFT matches for a classical baseline."""
    img0 = cv2.imread(img0_path, cv2.IMREAD_GRAYSCALE)
    img1 = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)
    if img0 is None:
        raise FileNotFoundError(f"Cannot read low-sun image: {img0_path}")
    if img1 is None:
        raise FileNotFoundError(f"Cannot read high-sun image: {img1_path}")

    sift = cv2.SIFT_create(nfeatures=2000)
    _, des0 = sift.detectAndCompute(img0, None)
    _, des1 = sift.detectAndCompute(img1, None)
    if des0 is None or des1 is None:
        return 0

    matches = cv2.BFMatcher().knnMatch(des0, des1, k=2)
    return sum(1 for first, second in matches if first.distance < 0.75 * second.distance)


def compute_sdi(pts: np.ndarray, img_shape: tuple[int, int], grid_divisions: int = 8) -> float:
    """
    Spatial Distribution Index (SDI) across an 8x8 grid:
        SDI = max(0.0, 1.0 - sigma_grid / (mu_grid + 1e-6))
    """
    if len(pts) == 0:
        return 0.0
    h, w = img_shape[:2]
    gh = h / float(grid_divisions)
    gw = w / float(grid_divisions)

    counts = np.zeros((grid_divisions, grid_divisions), dtype=np.float64)
    for x, y in pts:
        gx = min(max(int(x / gw), 0), grid_divisions - 1)
        gy = min(max(int(y / gh), 0), grid_divisions - 1)
        counts[gy, gx] += 1.0

    mu = float(np.mean(counts))
    sigma = float(np.std(counts))
    return round(float(max(0.0, 1.0 - (sigma / (mu + 1e-6)))), 4)


def compute_checkpoint_rmse(
    pts0_inliers: np.ndarray,
    pts1_inliers: np.ndarray,
    H: np.ndarray,
    split_ratio: float = 0.8,
    gsd_ref: float | None = None,
    random_seed: int = 42
) -> Dict[str, Any]:
    """
    80/20 Tie-Point / Check-Point validation split:
    - 80% Tie Points (used for geometric model fitting / verification)
    - 20% independent Check Points (held-out validation points)
    Computes RMSE in pixels and ground meters:
        RMSE_meters = RMSE_px * GSD_ref
    """
    n_inliers = len(pts0_inliers)
    if n_inliers < 5 or H is None:
        return {
            "checkpoint_count": 0,
            "tiepoint_count": n_inliers,
            "checkpoint_rmse_px": float("inf"),
            "checkpoint_rmse_meters": float("inf"),
            "tiepoint_rmse_px": float("inf"),
            "tiepoint_rmse_meters": float("inf"),
        }

    rng = np.random.RandomState(random_seed)
    indices = np.arange(n_inliers)
    rng.shuffle(indices)

    n_tie = max(int(round(split_ratio * n_inliers)), 4)
    tie_idx = indices[:n_tie]
    chk_idx = indices[n_tie:] if n_tie < n_inliers else indices[:max(1, n_inliers // 5)]

    def calc_reproj_rmse(p0: np.ndarray, p1: np.ndarray) -> float:
        if len(p0) == 0:
            return 0.0
        p0_proj = cv2.perspectiveTransform(
            p0.reshape(-1, 1, 2).astype(np.float32), H.astype(np.float64)
        ).reshape(-1, 2)
        err = np.linalg.norm(p0_proj - p1, axis=1)
        return float(np.sqrt(np.mean(err ** 2)))

    tie_rmse_px = calc_reproj_rmse(pts0_inliers[tie_idx], pts1_inliers[tie_idx])
    chk_rmse_px = calc_reproj_rmse(pts0_inliers[chk_idx], pts1_inliers[chk_idx]) if len(chk_idx) > 0 else tie_rmse_px

    gsd = float(gsd_ref) if gsd_ref is not None and gsd_ref > 0 else 1.0
    tie_rmse_m = tie_rmse_px * gsd
    chk_rmse_m = chk_rmse_px * gsd

    return {
        "checkpoint_count": len(chk_idx),
        "tiepoint_count": len(tie_idx),
        "checkpoint_rmse_px": round(chk_rmse_px, 4),
        "checkpoint_rmse_meters": round(chk_rmse_m, 4),
        "tiepoint_rmse_px": round(tie_rmse_px, 4),
        "tiepoint_rmse_meters": round(tie_rmse_m, 4),
    }

