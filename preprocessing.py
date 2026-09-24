"""
SunAngle/preprocessing.py
Planetary Image Radiometric and Illumination Preprocessing Engine
Designed for Chandrayaan-2 (OHRC, TMC, IIRS) and LRO NAC lunar imagery.
"""

import os
from pathlib import Path
from typing import Tuple, Optional, Union, Any, Dict
import cv2
import numpy as np


def load_lunar_image(img_input: Union[str, np.ndarray]) -> np.ndarray:
    """
    Loads an 8-bit, 16-bit, or floating point lunar image / GeoTIFF.
    Converts multi-channel to single-band grayscale and normalizes to float32 in [0, 1].
    Uses 1st to 99th percentile stretching to handle high dynamic range planetary sensors.
    """
    if isinstance(img_input, np.ndarray):
        img = img_input
    else:
        if not os.path.exists(img_input):
            raise FileNotFoundError(f"Lunar image file not found: {img_input}")
        img = cv2.imread(img_input, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise ValueError(f"Failed to decode image from path: {img_input}")

    # If multi-channel (RGB or multi-spectral band), convert or take mean/first band
    if img.ndim == 3:
        if img.shape[2] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        elif img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
        else:
            img = np.mean(img, axis=2)

    # Convert to float32
    img_f = img.astype(np.float32)

    # Planetary radiometric percentile stretch (handles deep shadow saturation & specular crater highlights)
    p_low, p_high = np.percentile(img_f, (1.0, 99.0))
    if p_high > p_low:
        img_stretched = np.clip((img_f - p_low) / (p_high - p_low), 0.0, 1.0)
    else:
        max_val = img_f.max()
        img_stretched = img_f / (max_val if max_val > 0 else 1.0)

    return img_stretched


def local_contrast_normalization(
    img_gray: np.ndarray,
    kernel_size: int = 71,
    eps: float = 1e-5
) -> np.ndarray:
    """
    Local Contrast Normalization (LCN) for single-scale illumination invariance.
    Removes low-frequency solar illumination gradients (high vs low sun angle)
    and equalizes local variance across lunar regolith.
    
    Formula: (I - local_mean) / (local_std + eps)
    """
    if img_gray.dtype != np.float32:
        img_gray = img_gray.astype(np.float32)
        if img_gray.max() > 1.0:
            img_gray /= 255.0

    if kernel_size % 2 == 0:
        kernel_size += 1

    # Fast local mean using box filter
    mean = cv2.boxFilter(img_gray, -1, (kernel_size, kernel_size), borderType=cv2.BORDER_REFLECT)
    
    # Local standard deviation: sqrt(E[X^2] - (E[X])^2)
    sq_mean = cv2.boxFilter(img_gray * img_gray, -1, (kernel_size, kernel_size), borderType=cv2.BORDER_REFLECT)
    var = np.maximum(sq_mean - mean * mean, 0.0)
    std = np.sqrt(var)

    # Normalize and clip dynamic range
    lcn = (img_gray - mean) / (std + eps)
    lcn = np.clip(lcn, -3.0, 3.0)
    # Map back to [0, 1]
    lcn_norm = (lcn + 3.0) / 6.0
    return lcn_norm.astype(np.float32)


def multi_scale_lcn(
    img_gray: np.ndarray,
    kernel_small: int = 25,
    kernel_large: int = 85,
    alpha: float = 0.5
) -> np.ndarray:
    """
    Multi-Scale Local Contrast Normalization (MS-LCN).
    Combines high-frequency edge equalization (micro-craters)
    with regional illumination removal (large crater topography and regional shadows).
    """
    lcn_s = local_contrast_normalization(img_gray, kernel_size=kernel_small)
    lcn_l = local_contrast_normalization(img_gray, kernel_size=kernel_large)
    ms_lcn = alpha * lcn_s + (1.0 - alpha) * lcn_l
    return np.clip(ms_lcn, 0.0, 1.0).astype(np.float32)


def normalized_gradient_field(
    img_gray: np.ndarray,
    eta: float = 0.01
) -> np.ndarray:
    """
    Computes Normalized Gradient Field (NGF) magnitude.
    Ideal for multi-modal registration (e.g. TMC panchromatic vs IIRS infrared).
    NGF is invariant to monotonic or non-linear radiometric transfer functions.
    """
    gx = cv2.Sobel(img_gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(img_gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx * gx + gy * gy)
    ngf = mag / (mag + eta)
    return ngf.astype(np.float32)


SENSOR_GSD = {
    "ohr": 0.25,
    "ohrc": 0.25,
    "tmc": 5.0,
    "tmc2": 5.0,
    "tmc-2": 5.0,
    "iirs": 80.0,
    "nac": 0.5,
    "lro": 0.5,
    "selene": 10.0,
    "kaguya": 10.0,
}


def infer_gsd_from_path(path_or_name: Union[str, Path, Any]) -> Optional[float]:
    """Infers Ground Sampling Distance (m/pixel) from sensor keywords in filename or path."""
    if not isinstance(path_or_name, (str, Path)):
        return None
    s = str(path_or_name).lower()
    for sensor, gsd in SENSOR_GSD.items():
        if sensor in s:
            return gsd
    return None


def apply_mtf_gaussian_filter(img: np.ndarray, scale_ratio: float) -> np.ndarray:
    """
    Applies Gaussian anti-aliasing filter modeling the sensor Modulation Transfer Function (MTF).
    Formula: sigma = sqrt(max((scale_ratio / 2)^2 - 0.5^2, 0.1))
    """
    scale_ratio = float(scale_ratio)
    val = (scale_ratio / 2.0) ** 2 - 0.5 ** 2
    sigma = float(np.sqrt(max(val, 0.1)))
    # Kernel size covering at least 3 sigmas on each side
    ksize = int(np.ceil(3.0 * sigma)) * 2 + 1
    ksize = max(ksize, 3)
    filtered = cv2.GaussianBlur(img, (ksize, ksize), sigmaX=sigma, sigmaY=sigma, borderType=cv2.BORDER_REFLECT)
    return filtered


def preprocess_lunar_pair(
    img0_input: Union[str, np.ndarray],
    img1_input: Union[str, np.ndarray],
    resize_long: int = 1024,
    use_multiscale: bool = True,
    lcn_kernel: int = 71
) -> Tuple[np.ndarray, np.ndarray, float, float]:
    """
    Loads and normalizes an image pair for deep feature matching.
    Returns:
        (img0_norm_uint8, img1_norm_uint8, scale0, scale1)
    where scale is the factor applied to resize the original image to resize_long.
    """
    img0 = load_lunar_image(img0_input)
    img1 = load_lunar_image(img1_input)

    if use_multiscale:
        img0_norm = multi_scale_lcn(img0, kernel_small=25, kernel_large=lcn_kernel)
        img1_norm = multi_scale_lcn(img1, kernel_small=25, kernel_large=lcn_kernel)
    else:
        img0_norm = local_contrast_normalization(img0, kernel_size=lcn_kernel)
        img1_norm = local_contrast_normalization(img1, kernel_size=lcn_kernel)

    # Convert to 8-bit uint8 [0, 255] for standard CV/deep learning input pipelines
    img0_u8 = (img0_norm * 255.0).astype(np.uint8)
    img1_u8 = (img1_norm * 255.0).astype(np.uint8)

    # Scale keeping aspect ratio
    def get_scaled(img: np.ndarray, target_long: int) -> Tuple[np.ndarray, float]:
        h, w = img.shape
        long_dim = max(h, w)
        if long_dim > target_long:
            scale = target_long / float(long_dim)
            new_w = int(round(w * scale))
            new_h = int(round(h * scale))
            # Ensure dimensions are divisible by 8 (requirement for LoFTR CNN feature maps)
            new_w = (new_w // 8) * 8
            new_h = (new_h // 8) * 8
            resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
            actual_scale_w = new_w / float(w)
            return resized, actual_scale_w
        else:
            # Ensure divisible by 8
            new_w = (w // 8) * 8
            new_h = (h // 8) * 8
            if new_w != w or new_h != h:
                resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
                return resized, new_w / float(w)
            return img, 1.0

    img0_scaled, scale0 = get_scaled(img0_u8, resize_long)
    img1_scaled, scale1 = get_scaled(img1_u8, resize_long)

    return img0_scaled, img1_scaled, scale0, scale1


def make_affine_transform(
    scale_x: float,
    scale_y: float,
    tx: float = 0.0,
    ty: float = 0.0
) -> np.ndarray:
    """Creates a 3x3 homogeneous affine transformation matrix."""
    return np.array([
        [float(scale_x), 0.0, float(tx)],
        [0.0, float(scale_y), float(ty)],
        [0.0, 0.0, 1.0]
    ], dtype=np.float64)


def apply_coordinate_transform(pts: np.ndarray, transform_matrix: np.ndarray) -> np.ndarray:
    """
    Applies a 3x3 affine or projective transformation matrix to (N, 2) coordinates.
    pts_out = (T @ [x, y, 1]^T)^T
    """
    if pts is None or len(pts) == 0:
        return np.empty((0, 2), dtype=np.float32)
    pts_arr = np.asarray(pts, dtype=np.float32)
    pts_h = np.column_stack([pts_arr, np.ones(len(pts_arr), dtype=np.float32)])
    transformed_h = (transform_matrix @ pts_h.T).T
    denom = transformed_h[:, 2:3]
    denom = np.where(np.abs(denom) < 1e-12, 1e-12, denom)
    return (transformed_h[:, :2] / denom).astype(np.float32)


def create_gsd_normalized_pair(
    img_src_input: Union[str, np.ndarray],
    img_ref_input: Union[str, np.ndarray],
    gsd_src: Optional[float] = None,
    gsd_ref: Optional[float] = None,
    target_common_gsd: Optional[float] = None,
    resize_long: int = 1024,
    use_multiscale: bool = True,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    GSD-Aware Physical Scale Normalization Layer for multi-resolution lunar imagery.
    
    Architecture:
    1. Determine a deterministic Common Physical GSD = target_common_gsd or max(gsd_src, gsd_ref).
    2. Convert both images to represent the exact same physical ground sampling distance:
       - If an image has finer resolution (gsd < common_gsd), downsample by (gsd / common_gsd)
         after applying sensor MTF Gaussian low-pass filtering.
    3. Construct explicit 3x3 affine coordinate transformation matrices for exact bi-directional
       mapping: Native <-> Common GSD <-> Scaled Matcher coordinates.
    4. Apply radiometric MS-LCN / LCN.
    5. Centralized matcher resizing: Rescale both common-GSD images by a SHARED scale factor,
       strictly preserving the 1:1 physical scale relationship established in Step 2.
    """
    # Infer GSD if not provided
    if gsd_src is None and isinstance(img_src_input, (str, Path)):
        gsd_src = infer_gsd_from_path(img_src_input)
    if gsd_ref is None and isinstance(img_ref_input, (str, Path)):
        gsd_ref = infer_gsd_from_path(img_ref_input)

    gsd_src_val = float(gsd_src) if gsd_src is not None and gsd_src > 0 else 1.0
    gsd_ref_val = float(gsd_ref) if gsd_ref is not None and gsd_ref > 0 else 1.0
    scale_ratio = gsd_ref_val / gsd_src_val  # s = GSD_ref / GSD_src

    # 1. Choose Common Physical GSD explicitly
    if target_common_gsd is not None and target_common_gsd > 0:
        common_gsd = float(target_common_gsd)
    else:
        common_gsd = max(gsd_src_val, gsd_ref_val)

    img_src_native = load_lunar_image(img_src_input)
    img_ref_native = load_lunar_image(img_ref_input)

    h_src_nat, w_src_nat = img_src_native.shape[:2]
    h_ref_nat, w_ref_nat = img_ref_native.shape[:2]

    # 2. Normalize both images to Common Physical GSD
    # Physical scale downsampling factor s_phys = gsd_image / common_gsd <= 1.0
    s_src_phys = gsd_src_val / common_gsd
    s_ref_phys = gsd_ref_val / common_gsd

    # Source normalization
    if s_src_phys < 0.98:
        inv_ratio = common_gsd / gsd_src_val
        src_blurred = apply_mtf_gaussian_filter(img_src_native, inv_ratio)
        new_w_src = max(int(round(w_src_nat * s_src_phys)), 16)
        new_h_src = max(int(round(h_src_nat * s_src_phys)), 16)
        img_src_common = cv2.resize(src_blurred, (new_w_src, new_h_src), interpolation=cv2.INTER_AREA)
        s_src_gsd_x = new_w_src / float(w_src_nat)
        s_src_gsd_y = new_h_src / float(h_src_nat)
    else:
        img_src_common = img_src_native.copy()
        s_src_gsd_x = 1.0
        s_src_gsd_y = 1.0

    # Reference normalization
    if s_ref_phys < 0.98:
        inv_ratio = common_gsd / gsd_ref_val
        ref_blurred = apply_mtf_gaussian_filter(img_ref_native, inv_ratio)
        new_w_ref = max(int(round(w_ref_nat * s_ref_phys)), 16)
        new_h_ref = max(int(round(h_ref_nat * s_ref_phys)), 16)
        img_ref_common = cv2.resize(ref_blurred, (new_w_ref, new_h_ref), interpolation=cv2.INTER_AREA)
        s_ref_gsd_x = new_w_ref / float(w_ref_nat)
        s_ref_gsd_y = new_h_ref / float(h_ref_nat)
    else:
        img_ref_common = img_ref_native.copy()
        s_ref_gsd_x = 1.0
        s_ref_gsd_y = 1.0

    # 3. Explicit affine transforms: Native <-> Common GSD
    T_src_nat_to_com = make_affine_transform(s_src_gsd_x, s_src_gsd_y)
    T_src_com_to_nat = make_affine_transform(1.0 / s_src_gsd_x, 1.0 / s_src_gsd_y)

    T_ref_nat_to_com = make_affine_transform(s_ref_gsd_x, s_ref_gsd_y)
    T_ref_com_to_nat = make_affine_transform(1.0 / s_ref_gsd_x, 1.0 / s_ref_gsd_y)

    # 4. Radiometric normalization (MS-LCN / LCN)
    if use_multiscale:
        src_norm = multi_scale_lcn(img_src_common, kernel_small=25, kernel_large=71)
        ref_norm = multi_scale_lcn(img_ref_common, kernel_small=25, kernel_large=71)
    else:
        src_norm = local_contrast_normalization(img_src_common, kernel_size=71)
        ref_norm = local_contrast_normalization(img_ref_common, kernel_size=71)

    src_u8 = (src_norm * 255.0).astype(np.uint8)
    ref_u8 = (ref_norm * 255.0).astype(np.uint8)

    # 5. Centralized matcher resizing preserving common physical scale
    # Apply a single shared scale factor so the physical GSD relationship remains 1:1
    h_src_c, w_src_c = src_u8.shape[:2]
    h_ref_c, w_ref_c = ref_u8.shape[:2]
    max_dim_common = max(h_src_c, w_src_c, h_ref_c, w_ref_c)

    if max_dim_common > resize_long:
        shared_scale = float(resize_long) / float(max_dim_common)
    else:
        shared_scale = 1.0

    def fit_matcher_div8(img: np.ndarray, scale: float) -> Tuple[np.ndarray, float, float]:
        h, w = img.shape[:2]
        new_w = max((int(round(w * scale)) // 8) * 8, 64)
        new_h = max((int(round(h * scale)) // 8) * 8, 64)
        if new_w != w or new_h != h:
            interp = cv2.INTER_AREA if (new_w < w or new_h < h) else cv2.INTER_CUBIC
            resized = cv2.resize(img, (new_w, new_h), interpolation=interp)
            return resized, new_w / float(w), new_h / float(h)
        return img.copy(), 1.0, 1.0

    src_scaled, s_src_m_x, s_src_m_y = fit_matcher_div8(src_u8, shared_scale)
    ref_scaled, s_ref_m_x, s_ref_m_y = fit_matcher_div8(ref_u8, shared_scale)

    # Common <-> Matcher transforms
    T_src_com_to_match = make_affine_transform(s_src_m_x, s_src_m_y)
    T_src_match_to_com = make_affine_transform(1.0 / s_src_m_x, 1.0 / s_src_m_y)

    T_ref_com_to_match = make_affine_transform(s_ref_m_x, s_ref_m_y)
    T_ref_match_to_com = make_affine_transform(1.0 / s_ref_m_x, 1.0 / s_ref_m_y)

    # 6. Cumulative full affine transforms: Native <-> Matcher
    T_src_nat_to_match = T_src_com_to_match @ T_src_nat_to_com
    T_src_match_to_nat = T_src_com_to_nat @ T_src_match_to_com

    T_ref_nat_to_match = T_ref_com_to_match @ T_ref_nat_to_com
    T_ref_match_to_nat = T_ref_com_to_nat @ T_ref_match_to_com

    s_src_total_x = float(T_src_nat_to_match[0, 0])
    s_ref_total_x = float(T_ref_nat_to_match[0, 0])

    if verbose and (scale_ratio > 1.2 or scale_ratio < 0.8):
        print(f"[GSD-NORMALIZATION] GSD src={gsd_src_val:.3f}m, ref={gsd_ref_val:.3f}m | Common GSD={common_gsd:.3f}m")
        print(f"[GSD-NORMALIZATION] Native shapes: src=({h_src_nat},{w_src_nat}), ref=({h_ref_nat},{w_ref_nat})")
        print(f"[GSD-NORMALIZATION] Common shapes: src=({h_src_c},{w_src_c}), ref=({h_ref_c},{w_ref_c})")
        print(f"[GSD-NORMALIZATION] Matcher shapes: src={src_scaled.shape}, ref={ref_scaled.shape} (scale_src={s_src_total_x:.4f}, scale_ref={s_ref_total_x:.4f})")

    return {
        "img_src_scaled": src_scaled,
        "img_ref_scaled": ref_scaled,
        "img_src_common": img_src_common,
        "img_ref_common": img_ref_common,
        "img_src_native": img_src_native,
        "img_ref_native": img_ref_native,
        "T_src_native_to_matcher": T_src_nat_to_match,
        "T_src_matcher_to_native": T_src_match_to_nat,
        "T_ref_native_to_matcher": T_ref_nat_to_match,
        "T_ref_matcher_to_native": T_ref_match_to_nat,
        "T_src_native_to_common": T_src_nat_to_com,
        "T_src_common_to_native": T_src_com_to_nat,
        "T_ref_native_to_common": T_ref_nat_to_com,
        "T_ref_common_to_native": T_ref_com_to_nat,
        "T_src_common_to_matcher": T_src_com_to_match,
        "T_src_matcher_to_common": T_src_match_to_com,
        "T_ref_common_to_matcher": T_ref_com_to_match,
        "T_ref_matcher_to_common": T_ref_match_to_com,
        "scale_src_total": s_src_total_x,
        "scale_ref_total": s_ref_total_x,
        "scale_src_gsd": s_src_phys,
        "scale_ref_gsd": s_ref_phys,
        "scale_ratio": scale_ratio,
        "common_gsd": common_gsd,
        "gsd_src": gsd_src_val,
        "gsd_ref": gsd_ref_val,
    }


def preprocess_cross_sensor_pair(
    img_ref_input: Union[str, np.ndarray],
    img_target_input: Union[str, np.ndarray],
    sensor_ref: Optional[str] = None,
    sensor_target: Optional[str] = None,
    gsd_ref: Optional[float] = None,
    gsd_target: Optional[float] = None,
    resize_long: int = 1024
) -> Tuple[np.ndarray, np.ndarray, float, float]:
    """
    Convenience wrapper for cross-sensor pairs.
    Returns:
        (ref_scaled, target_scaled, scale_ref_total, scale_target_total)
    """
    if gsd_ref is None and sensor_ref:
        gsd_ref = SENSOR_GSD.get(sensor_ref.lower(), None)
    if gsd_target is None and sensor_target:
        gsd_target = SENSOR_GSD.get(sensor_target.lower(), None)

    res = create_gsd_normalized_pair(
        img_src_input=img_target_input,
        img_ref_input=img_ref_input,
        gsd_src=gsd_target,
        gsd_ref=gsd_ref,
        resize_long=resize_long
    )
    return res["img_ref_scaled"], res["img_src_scaled"], res["scale_ref_total"], res["scale_src_total"]

