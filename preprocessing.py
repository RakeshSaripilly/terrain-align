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
    "ohrc": 0.25,
    "tmc": 5.0,
    "tmc2": 5.0,
    "tmc-2": 5.0,
    "iirs": 80.0,
    "nac": 0.5,
    "lro": 0.5,
    "selene": 10.0,
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


def create_gsd_normalized_pair(
    img_src_input: Union[str, np.ndarray],
    img_ref_input: Union[str, np.ndarray],
    gsd_src: Optional[float] = None,
    gsd_ref: Optional[float] = None,
    resize_long: int = 1024,
    use_multiscale: bool = True
) -> Dict[str, Any]:
    """
    GSD-Aware Scale Normalization Layer for multi-resolution lunar imagery.
    
    Computes scale ratio s = GSD_ref / GSD_src.
    - If s > 1.05: Source has higher resolution (smaller GSD in m/px).
      Convolves source with MTF Gaussian blur and downsamples by 1/s via INTER_AREA.
    - If s < 0.95: Reference has higher resolution.
      Convolves reference with MTF Gaussian blur and downsamples by s via INTER_AREA.
    - Then applies radiometric MS-LCN and resizes to resize_long (divisible by 8).
    
    Returns dictionary with virtual coarse images, native full-resolution images,
    and cumulative coordinate scale factors for accurate back-projection.
    """
    # Infer GSD if not provided
    if gsd_src is None and isinstance(img_src_input, (str, Path)):
        gsd_src = infer_gsd_from_path(img_src_input)
    if gsd_ref is None and isinstance(img_ref_input, (str, Path)):
        gsd_ref = infer_gsd_from_path(img_ref_input)

    gsd_src_val = float(gsd_src) if gsd_src is not None and gsd_src > 0 else 1.0
    gsd_ref_val = float(gsd_ref) if gsd_ref is not None and gsd_ref > 0 else 1.0
    scale_ratio = gsd_ref_val / gsd_src_val  # s = GSD_ref / GSD_src

    img_src_native = load_lunar_image(img_src_input)
    img_ref_native = load_lunar_image(img_ref_input)

    # Initial downsample factor before LoFTR resize
    s_src_gsd = 1.0
    s_ref_gsd = 1.0

    if scale_ratio > 1.05:
        # Source has higher resolution -> downsample source to match reference GSD
        src_blurred = apply_mtf_gaussian_filter(img_src_native, scale_ratio)
        new_w = max(int(round(img_src_native.shape[1] / scale_ratio)), 16)
        new_h = max(int(round(img_src_native.shape[0] / scale_ratio)), 16)
        img_src_virtual = cv2.resize(src_blurred, (new_w, new_h), interpolation=cv2.INTER_AREA)
        s_src_gsd = new_w / float(img_src_native.shape[1])
        img_ref_virtual = img_ref_native.copy()
    elif scale_ratio < 0.95:
        # Reference has higher resolution -> downsample reference to match source GSD
        inv_ratio = 1.0 / scale_ratio
        ref_blurred = apply_mtf_gaussian_filter(img_ref_native, inv_ratio)
        new_w = max(int(round(img_ref_native.shape[1] * scale_ratio)), 16)
        new_h = max(int(round(img_ref_native.shape[0] * scale_ratio)), 16)
        img_ref_virtual = cv2.resize(ref_blurred, (new_w, new_h), interpolation=cv2.INTER_AREA)
        s_ref_gsd = new_w / float(img_ref_native.shape[1])
        img_src_virtual = img_src_native.copy()
    else:
        img_src_virtual = img_src_native.copy()
        img_ref_virtual = img_ref_native.copy()

    # Preprocess with MS-LCN
    if use_multiscale:
        src_norm = multi_scale_lcn(img_src_virtual, kernel_small=25, kernel_large=71)
        ref_norm = multi_scale_lcn(img_ref_virtual, kernel_small=25, kernel_large=71)
    else:
        src_norm = local_contrast_normalization(img_src_virtual, kernel_size=71)
        ref_norm = local_contrast_normalization(img_ref_virtual, kernel_size=71)

    src_u8 = (src_norm * 255.0).astype(np.uint8)
    ref_u8 = (ref_norm * 255.0).astype(np.uint8)

    # Scale to resize_long divisible by 8 for LoFTR
    def scale_div8(img: np.ndarray, target_long: int) -> Tuple[np.ndarray, float]:
        h, w = img.shape[:2]
        long_dim = max(h, w)
        if long_dim > target_long:
            scale = target_long / float(long_dim)
            new_w = (int(round(w * scale)) // 8) * 8
            new_h = (int(round(h * scale)) // 8) * 8
            new_w = max(new_w, 64)
            new_h = max(new_h, 64)
            resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
            return resized, new_w / float(w)
        else:
            new_w = (w // 8) * 8
            new_h = (h // 8) * 8
            new_w = max(new_w, 64)
            new_h = max(new_h, 64)
            if new_w != w or new_h != h:
                resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
                return resized, new_w / float(w)
            return img, 1.0

    src_scaled, s_src_loftr = scale_div8(src_u8, resize_long)
    ref_scaled, s_ref_loftr = scale_div8(ref_u8, resize_long)

    # Total scale factor from native image to scaled matcher input:
    # x_scaled = x_native * s_total  =>  x_native = x_scaled / s_total
    s_src_total = s_src_gsd * s_src_loftr
    s_ref_total = s_ref_gsd * s_ref_loftr

    return {
        "img_src_scaled": src_scaled,
        "img_ref_scaled": ref_scaled,
        "scale_src_total": s_src_total,
        "scale_ref_total": s_ref_total,
        "scale_src_gsd": s_src_gsd,
        "scale_ref_gsd": s_ref_gsd,
        "scale_ratio": scale_ratio,
        "gsd_src": gsd_src_val,
        "gsd_ref": gsd_ref_val,
        "img_src_native": img_src_native,
        "img_ref_native": img_ref_native,
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

