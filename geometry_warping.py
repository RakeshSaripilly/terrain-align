"""
SunAngle/geometry_warping.py
Geometric Modeling, Outlier Rejection, and Sub-Pixel Terrain Warping Engine
Computes USAC_MAGSAC Homography and Thin-Plate Spline (TPS) transformations for planetary relief.
"""

from typing import Tuple, Dict, Any, Optional
import cv2
import numpy as np


def compute_reprojection_rmse(
    pts0: np.ndarray,
    pts1: np.ndarray,
    H: np.ndarray
) -> Tuple[float, np.ndarray]:
    """
    Computes bidirectional and forward reprojection RMSE between matched points.

    Args:
        pts0: (N, 2) reference points
        pts1: (N, 2) target points
        H: (3, 3) homography mapping pts0 -> pts1

    Returns:
        rmse: float Root Mean Square Error in pixels
        residuals: (N,) per-point Euclidean transfer error
    """
    if len(pts0) == 0 or H is None:
        return float('inf'), np.array([])

    pts0_h = np.column_stack([pts0, np.ones(len(pts0))])  # (N, 3)
    pred1_h = (H @ pts0_h.T).T
    pred1 = pred1_h[:, :2] / (pred1_h[:, 2:3] + 1e-12)

    diff = pts1 - pred1
    residuals = np.linalg.norm(diff, axis=1)
    rmse = float(np.sqrt(np.mean(residuals * residuals)))
    return round(rmse, 4), residuals


def estimate_robust_transformation(
    pts0: np.ndarray,
    pts1: np.ndarray,
    ransac_thresh: float = 2.0,
    max_iters: int = 15000,
    confidence: float = 0.999
) -> Dict[str, Any]:
    """
    Robust geometric estimation using USAC_MAGSAC (or RANSAC fallback).
    Computes homography, inlier mask, and precision RMSE.
    """
    if len(pts0) < 4:
        return {
            "success": False,
            "H": None,
            "inlier_mask": np.zeros(len(pts0), dtype=bool),
            "inlier_count": 0,
            "inlier_ratio": 0.0,
            "rmse": float('inf'),
            "residuals": np.array([])
        }

    # Attempt USAC_MAGSAC with fallback to standard RANSAC
    try:
        H, mask = cv2.findHomography(
            pts0, pts1,
            method=cv2.USAC_MAGSAC,
            ransacReprojThreshold=ransac_thresh,
            maxIters=max_iters,
            confidence=confidence
        )
    except Exception:
        H, mask = cv2.findHomography(
            pts0, pts1,
            method=cv2.RANSAC,
            ransacReprojThreshold=ransac_thresh,
            maxIters=max_iters
        )

    if H is None or mask is None:
        return {
            "success": False,
            "H": None,
            "inlier_mask": np.zeros(len(pts0), dtype=bool),
            "inlier_count": 0,
            "inlier_ratio": 0.0,
            "rmse": float('inf'),
            "residuals": np.array([])
        }

    inlier_mask = mask.ravel().astype(bool)
    inlier_count = int(np.sum(inlier_mask))
    inlier_ratio = float(inlier_count / len(pts0)) if len(pts0) > 0 else 0.0

    inlier_pts0 = pts0[inlier_mask]
    inlier_pts1 = pts1[inlier_mask]
    rmse, residuals = compute_reprojection_rmse(inlier_pts0, inlier_pts1, H)

    return {
        "success": True,
        "H": H,
        "inlier_mask": inlier_mask,
        "inlier_count": inlier_count,
        "inlier_ratio": round(inlier_ratio, 4),
        "rmse": rmse,
        "residuals": residuals,
        "inlier_pts0": inlier_pts0,
        "inlier_pts1": inlier_pts1
    }


def warp_lunar_image(
    img_to_warp: np.ndarray,
    ref_shape: Tuple[int, int],
    H: np.ndarray,
    interpolation: int = cv2.INTER_CUBIC
) -> np.ndarray:
    """
    Warps target image onto reference image coordinate frame using homography H.
    """
    h_ref, w_ref = ref_shape[:2]
    # H maps pts0 (ref) -> pts1 (target). To warp target to ref, cv2.warpPerspective requires H_inv:
    # However, if H was computed with findHomography(pts_target, pts_ref), then H maps target -> ref.
    warped = cv2.warpPerspective(img_to_warp, H, (w_ref, h_ref), flags=interpolation, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return warped


class ThinPlateSplineWarp:
    """
    Thin-Plate Spline (TPS) transformation using SciPy RBFInterpolator.
    Captures non-rigid parallax displacements caused by crater topography and high-elevation rims.
    """

    def __init__(self, pts_src: np.ndarray, pts_dst: np.ndarray, smoothing: float = 0.5):
        from scipy.interpolate import RBFInterpolator
        self.pts_src = pts_src.astype(np.float64)
        self.pts_dst = pts_dst.astype(np.float64)
        # Fit RBF: mapping from src to dst displacements
        disp = self.pts_dst - self.pts_src
        self.rbf = RBFInterpolator(self.pts_src, disp, kernel='thin_plate_spline', smoothing=smoothing)

    def transform_points(self, pts: np.ndarray) -> np.ndarray:
        pts = pts.astype(np.float64)
        disp = self.rbf(pts)
        return pts + disp

    def warp_image(
        self,
        img: np.ndarray,
        ref_shape: Tuple[int, int],
        grid_step: int = 16
    ) -> np.ndarray:
        h_ref, w_ref = ref_shape[:2]
        # Coarse-to-fine displacement grid for fast dense warping
        y_c = np.arange(0, h_ref, grid_step)
        x_c = np.arange(0, w_ref, grid_step)
        if y_c[-1] != h_ref - 1:
            y_c = np.append(y_c, h_ref - 1)
        if x_c[-1] != w_ref - 1:
            x_c = np.append(x_c, w_ref - 1)

        xx_c, yy_c = np.meshgrid(x_c, y_c)
        grid_pts = np.column_stack([xx_c.ravel(), yy_c.ravel()])
        # Inverse mapping: from ref coordinates to target coordinates
        mapped_pts = self.transform_points(grid_pts)
        map_x_c = mapped_pts[:, 0].reshape(len(y_c), len(x_c)).astype(np.float32)
        map_y_c = mapped_pts[:, 1].reshape(len(y_c), len(x_c)).astype(np.float32)

        # Upsample map to full resolution
        map_x = cv2.resize(map_x_c, (w_ref, h_ref), interpolation=cv2.INTER_CUBIC)
        map_y = cv2.resize(map_y_c, (w_ref, h_ref), interpolation=cv2.INTER_CUBIC)

        warped = cv2.remap(img, map_x, map_y, interpolation=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        return warped


def fit_thin_plate_spline(
    pts_src: np.ndarray,
    pts_dst: np.ndarray,
    max_control_pts: int = 120,
    smoothing: float = 0.5
) -> Optional[ThinPlateSplineWarp]:
    """
    Fits a Thin-Plate Spline (TPS) transformation.
    Captures non-linear local parallax caused by high crater rims and rough lunar topography.
    """
    if len(pts_src) < 6:
        return None

    if len(pts_src) > max_control_pts:
        step = len(pts_src) // max_control_pts
        pts_src = pts_src[::step][:max_control_pts]
        pts_dst = pts_dst[::step][:max_control_pts]

    try:
        tps = ThinPlateSplineWarp(pts_src, pts_dst, smoothing=smoothing)
        return tps
    except Exception:
        return None



def generate_checkerboard(
    img0: np.ndarray,
    img1_warped: np.ndarray,
    square_size: int = 64
) -> np.ndarray:
    """
    Creates a checkerboard overlay between reference image and warped target image.
    Enables immediate visual inspection of crater rim and ridge alignment continuity.
    """
    h, w = img0.shape[:2]
    if img1_warped.shape[:2] != (h, w):
        img1_warped = cv2.resize(img1_warped, (w, h))

    # Convert both to 3-channel if needed
    def ensure_3ch(img):
        if img.ndim == 2:
            return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        return img

    c0 = ensure_3ch(img0)
    c1 = ensure_3ch(img1_warped)

    y_indices, x_indices = np.indices((h, w))
    checkers = ((x_indices // square_size) + (y_indices // square_size)) % 2 == 0

    composite = np.zeros_like(c0)
    composite[checkers] = c0[checkers]
    composite[~checkers] = c1[~checkers]
    return composite


def compute_difference_map(
    img0: np.ndarray,
    img1_warped: np.ndarray,
    valid_mask: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Computes normalized absolute difference and false-color heatmap of alignment.
    Dark blue = perfect sub-pixel alignment, Red/Yellow = residual discrepancy.
    """
    h, w = img0.shape[:2]
    if img1_warped.shape[:2] != (h, w):
        img1_warped = cv2.resize(img1_warped, (w, h))

    g0 = img0.astype(np.float32) if img0.ndim == 2 else cv2.cvtColor(img0, cv2.COLOR_BGR2GRAY).astype(np.float32)
    g1 = img1_warped.astype(np.float32) if img1_warped.ndim == 2 else cv2.cvtColor(img1_warped, cv2.COLOR_BGR2GRAY).astype(np.float32)

    if g0.max() > 1.0:
        g0 /= 255.0
    if g1.max() > 1.0:
        g1 /= 255.0

    # Ignore zero-padded warped borders
    mask = (g1 > 0.01) if valid_mask is None else valid_mask

    diff = np.abs(g0 - g1)
    diff[~mask] = 0.0

    # Normalize difference for visualization
    diff_u8 = np.clip(diff * 255.0 * 2.0, 0, 255).astype(np.uint8)
    heatmap = cv2.applyColorMap(diff_u8, cv2.COLORMAP_JET)
    heatmap[~mask] = 0

    return diff, heatmap
