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
