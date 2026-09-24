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


def compute_scale_consistency(
    matrix: np.ndarray,
    expected_scale_ratio: float = 1.0,
    eval_point: Optional[Tuple[float, float]] = None,
    tolerance: float = 0.20,
    is_normalized_space: bool = False
) -> Dict[str, Any]:
    """
    Scale Consistency Check:
    Verifies that the determinant of the estimated local transformation matrix |J|
    matches the expected scale factor squared (s^2 +- tolerance).
    
    In common-GSD / normalized space, expected_scale_ratio is 1.0 (residual scale).
    In native sensor space, expected_scale_ratio is GSD_ref / GSD_src.
    
    Args:
        matrix: (2, 3) affine matrix or (3, 3) homography
        expected_scale_ratio: s = scale factor (default 1.0 for normalized coordinates)
        eval_point: Point (x, y) at which to evaluate Jacobian (evaluates at affine center if None)
        tolerance: Allowed fractional deviation (default 20% = 0.20)
        is_normalized_space: Whether evaluation is in normalized matcher space
    """
    if matrix is None:
        return {"scale_consistency_pass": False, "det_J": 0.0, "expected_scale_sq": 0.0, "scale_error_ratio": 1.0}

    s_expected_sq = float(expected_scale_ratio ** 2)

    if matrix.shape == (2, 3):
        # Affine matrix [ [a1, a2, a0], [b1, b2, b0] ]
        det_J = abs(float(matrix[0, 0] * matrix[1, 1] - matrix[0, 1] * matrix[1, 0]))
    elif matrix.shape == (3, 3):
        # Homography: local Jacobian determinant
        if eval_point is None:
            # Canonical center where projective denominator den = h33 = 1.0
            det_J = abs(float(matrix[0, 0] * matrix[1, 1] - matrix[0, 1] * matrix[1, 0]))
        else:
            x, y = float(eval_point[0]), float(eval_point[1])
            h11, h12, h13 = matrix[0]
            h21, h22, h23 = matrix[1]
            h31, h32, h33 = matrix[2]
            den = h31 * x + h32 * y + h33
            if abs(den) < 1e-8:
                det_J = abs(float(np.linalg.det(matrix[:2, :2])))
            else:
                num_x = h11 * x + h12 * y + h13
                num_y = h21 * x + h22 * y + h23
                dx_dx = (h11 * den - num_x * h31) / (den * den)
                dx_dy = (h12 * den - num_x * h32) / (den * den)
                dy_dx = (h21 * den - num_y * h31) / (den * den)
                dy_dy = (h22 * den - num_y * h32) / (den * den)
                det_J = abs(float(dx_dx * dy_dy - dx_dy * dy_dx))
    else:
        det_J = 1.0

    if s_expected_sq > 0:
        error_ratio = abs(det_J - s_expected_sq) / s_expected_sq
        passed = bool(error_ratio <= tolerance)
    else:
        error_ratio = 0.0
        passed = True

    return {
        "scale_consistency_pass": passed,
        "det_J": round(float(det_J), 4),
        "expected_scale_sq": round(float(s_expected_sq), 4),
        "scale_error_ratio": round(float(error_ratio), 4),
        "tolerance": tolerance,
        "is_normalized_space": is_normalized_space
    }


def estimate_robust_transformation(
    pts0: np.ndarray,
    pts1: np.ndarray,
    ransac_thresh: float = 2.0,
    max_iters: int = 15000,
    confidence: float = 0.999,
    model: str = "homography",
    gsd_ratio: float = 1.0,
    expected_residual_scale: float = 1.0
) -> Dict[str, Any]:
    """
    Robust geometric estimation using USAC_MAGSAC (or RANSAC fallback).
    Supports homography, affine, and Thin Plate Spline (TPS) transformation models.
    Scales residual threshold adaptively by GSD scale ratio.
    """
    pts0 = np.asarray(pts0, dtype=np.float32)
    pts1 = np.asarray(pts1, dtype=np.float32)

    min_pts_required = 3 if model.lower() == "affine" else 4
    if len(pts0) < min_pts_required:
        return {
            "success": False,
            "H": None,
            "M": None,
            "tps": None,
            "model_type": model,
            "inlier_mask": np.zeros(len(pts0), dtype=bool),
            "inlier_count": 0,
            "inlier_ratio": 0.0,
            "rmse": float('inf'),
            "residuals": np.array([])
        }

    # Adaptive threshold scaled by GSD scale ratio
    ratio = float(gsd_ratio) if gsd_ratio > 0 else 1.0
    scale_factor = max(1.0, min(ratio, 1.0 / ratio) ** 0.5) if (ratio > 1.05 or ratio < 0.95) else 1.0
    adaptive_thresh = float(ransac_thresh * scale_factor)

    model_lower = model.lower()
    H = None
    M = None
    mask = None
    tps = None

    if model_lower == "affine":
        # Estimate Affine transformation (prevents shear distortion across planetary scales)
        try:
            M, inliers = cv2.estimateAffine2D(
                pts0, pts1,
                method=cv2.USAC_MAGSAC,
                ransacReprojThreshold=adaptive_thresh,
                maxIters=max_iters,
                confidence=confidence
            )
            mask = inliers
        except Exception:
            try:
                M, inliers = cv2.estimateAffinePartial2D(
                    pts0, pts1,
                    method=cv2.RANSAC,
                    ransacReprojThreshold=adaptive_thresh,
                    maxIters=max_iters
                )
                mask = inliers
            except Exception:
                M, mask = None, None

        if M is not None:
            # Construct 3x3 homogeneous matrix representation for unified evaluation
            H = np.vstack([M, [0.0, 0.0, 1.0]])

    else:
        # Default Homography (USAC_MAGSAC with standard RANSAC fallback)
        try:
            H, mask = cv2.findHomography(
                pts0, pts1,
                method=cv2.USAC_MAGSAC,
                ransacReprojThreshold=adaptive_thresh,
                maxIters=max_iters,
                confidence=confidence
            )
        except Exception:
            H, mask = cv2.findHomography(
                pts0, pts1,
                method=cv2.RANSAC,
                ransacReprojThreshold=adaptive_thresh,
                maxIters=max_iters
            )
        if H is not None:
            M = H[:2, :]

    if H is None or mask is None:
        return {
            "success": False,
            "H": None,
            "M": None,
            "tps": None,
            "model_type": model,
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

    # If model is TPS, fit Thin Plate Spline on geometrically verified inliers
    if model_lower == "tps" and inlier_count >= 6:
        tps = fit_thin_plate_spline(inlier_pts0, inlier_pts1)

    # Centroid of inliers for evaluating local Jacobian determinant
    eval_pt = tuple(np.mean(inlier_pts0, axis=0)) if inlier_count > 0 else (0.0, 0.0)

    # Scale consistency check on residual transformation in current coordinate space
    # Residual scale between normalized images is expected to be ~1.0
    scale_consistency = compute_scale_consistency(
        H,
        expected_scale_ratio=expected_residual_scale,
        eval_point=eval_pt,
        is_normalized_space=(expected_residual_scale == 1.0)
    )
    scale_consistency["gsd_ratio"] = gsd_ratio

    return {
        "success": True,
        "H": H,
        "M": M,
        "tps": tps,
        "model_type": model,
        "inlier_mask": inlier_mask,
        "inlier_count": inlier_count,
        "inlier_ratio": round(inlier_ratio, 4),
        "rmse": rmse,
        "residuals": residuals,
        "inlier_pts0": inlier_pts0,
        "inlier_pts1": inlier_pts1,
        "adaptive_thresh": adaptive_thresh,
        "scale_consistency": scale_consistency
    }


def warp_lunar_image_memory_safe(
    img_to_warp: np.ndarray,
    ref_shape: Tuple[int, int],
    H_or_M: np.ndarray,
    is_affine: bool = False,
    interpolation: int = cv2.INTER_CUBIC,
    max_tile_size: int = 4096
) -> np.ndarray:
    """
    Memory-safe dual warping for large lunar canvases (> 4096 x 4096 px).
    Prevents Out-Of-Memory (OOM) crashes by tiling large arrays during warpPerspective / warpAffine.
    """
    h_ref, w_ref = ref_shape[:2]

    # Standard direct warping if dimensions within safe threshold
    if h_ref <= max_tile_size and w_ref <= max_tile_size:
        if is_affine or H_or_M.shape == (2, 3):
            M = H_or_M[:2, :]
            return cv2.warpAffine(img_to_warp, M, (w_ref, h_ref), flags=interpolation, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        else:
            return cv2.warpPerspective(img_to_warp, H_or_M, (w_ref, h_ref), flags=interpolation, borderMode=cv2.BORDER_CONSTANT, borderValue=0)

    # Tiled memory-safe warping
    tile_size = 2048
    out_shape = (h_ref, w_ref) if img_to_warp.ndim == 2 else (h_ref, w_ref, img_to_warp.shape[2])
    warped_full = np.zeros(out_shape, dtype=img_to_warp.dtype)

    for y0 in range(0, h_ref, tile_size):
        y1 = min(y0 + tile_size, h_ref)
        tile_h = y1 - y0
        for x0 in range(0, w_ref, tile_size):
            x1 = min(x0 + tile_size, w_ref)
            tile_w = x1 - x0

            # Translation matrix shifting tile origin to (0, 0)
            # T_shift * H maps coordinates so (x0, y0) is at (0, 0) in tile
            T_shift = np.array([
                [1.0, 0.0, -float(x0)],
                [0.0, 1.0, -float(y0)],
                [0.0, 0.0, 1.0]
            ], dtype=np.float64)

            if is_affine or H_or_M.shape == (2, 3):
                H_3x3 = np.vstack([H_or_M[:2, :], [0.0, 0.0, 1.0]])
                tile_H = T_shift @ H_3x3
                tile_warp = cv2.warpAffine(
                    img_to_warp, tile_H[:2, :], (tile_w, tile_h),
                    flags=interpolation, borderMode=cv2.BORDER_CONSTANT, borderValue=0
                )
            else:
                tile_H = T_shift @ H_or_M
                tile_warp = cv2.warpPerspective(
                    img_to_warp, tile_H, (tile_w, tile_h),
                    flags=interpolation, borderMode=cv2.BORDER_CONSTANT, borderValue=0
                )

            warped_full[y0:y1, x0:x1] = tile_warp

    return warped_full


def warp_lunar_image(
    img_to_warp: np.ndarray,
    ref_shape: Tuple[int, int],
    H_or_tps: Any,
    interpolation: int = cv2.INTER_CUBIC
) -> np.ndarray:
    """
    Warps target image onto reference image coordinate frame using homography H, Affine M,
    or ThinPlateSplineWarp object.
    """
    if H_or_tps is None:
        return np.zeros(ref_shape[:2], dtype=img_to_warp.dtype)
    if hasattr(H_or_tps, "warp_image"):
        return H_or_tps.warp_image(img_to_warp, ref_shape)
    return warp_lunar_image_memory_safe(img_to_warp, ref_shape, H_or_tps, is_affine=False, interpolation=interpolation)



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
        if h_ref * w_ref <= 512 * 512:
            # Direct exact dense evaluation for standard / medium canvases
            xx, yy = np.meshgrid(np.arange(w_ref, dtype=np.float32), np.arange(h_ref, dtype=np.float32))
            grid = np.column_stack([xx.ravel(), yy.ravel()])
            mapped = self.transform_points(grid)
            map_x = mapped[:, 0].reshape(h_ref, w_ref).astype(np.float32)
            map_y = mapped[:, 1].reshape(h_ref, w_ref).astype(np.float32)
        else:
            # Uniform linspace displacement grid for large canvases (> 512x512)
            steps_y = min(h_ref, max(16, h_ref // grid_step))
            steps_x = min(w_ref, max(16, w_ref // grid_step))
            y_c = np.linspace(0, h_ref - 1, steps_y, dtype=np.float64)
            x_c = np.linspace(0, w_ref - 1, steps_x, dtype=np.float64)
            xx_c, yy_c = np.meshgrid(x_c, y_c)
            grid_pts = np.column_stack([xx_c.ravel(), yy_c.ravel()])
            disp = self.rbf(grid_pts)
            disp_x = disp[:, 0].reshape(steps_y, steps_x).astype(np.float32)
            disp_y = disp[:, 1].reshape(steps_y, steps_x).astype(np.float32)
            disp_x_full = cv2.resize(disp_x, (w_ref, h_ref), interpolation=cv2.INTER_LINEAR)
            disp_y_full = cv2.resize(disp_y, (w_ref, h_ref), interpolation=cv2.INTER_LINEAR)
            xx_full, yy_full = np.meshgrid(np.arange(w_ref, dtype=np.float32), np.arange(h_ref, dtype=np.float32))
            map_x = xx_full + disp_x_full
            map_y = yy_full + disp_y_full

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
