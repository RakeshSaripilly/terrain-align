"""
SunAngle/matching_engine.py
Deep Transformer Feature Extraction and Scale-Space Matching Engine
Multi-modal and scale-invariant correspondence engine for Chandrayaan-2 (OHRC, TMC, IIRS).
"""

from typing import Dict, Any, Tuple, Optional, List
import ssl
import cv2
import numpy as np
import torch

ssl._create_default_https_context = ssl._create_unverified_context

from kornia.feature import LoFTR
try:
    from .preprocessing import preprocess_lunar_pair, local_contrast_normalization
except (ImportError, ValueError):
    try:
        from preprocessing import preprocess_lunar_pair, local_contrast_normalization
    except ImportError:
        from SunAngle.preprocessing import preprocess_lunar_pair, local_contrast_normalization


class LunarLoFTRMatcher:
    """
    Planetary Deep Feature Correspondence Engine.
    Uses transformer-based coarse-to-fine matching resilient to extreme sun angle variations.
    """

    def __init__(self, pretrained: str = 'outdoor', device: Optional[str] = None):
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        self.matcher = LoFTR(pretrained=pretrained).to(self.device).eval()

    @torch.no_grad()
    def match(
        self,
        img0_norm_u8: np.ndarray,
        img1_norm_u8: np.ndarray,
        conf_thresh: float = 0.2
    ) -> Dict[str, np.ndarray]:
        """
        Runs LoFTR forward pass on normalized grayscale pairs.
        Both images must have height and width divisible by 8.
        """
        # Ensure divisible by 8
        h0, w0 = img0_norm_u8.shape[:2]
        h1, w1 = img1_norm_u8.shape[:2]

        def prepare_tensor(img: np.ndarray) -> torch.Tensor:
            t = torch.from_numpy(img).float() / 255.0
            return t[None, None].to(self.device)

        t0 = prepare_tensor(img0_norm_u8)
        t1 = prepare_tensor(img1_norm_u8)

        input_dict = {"image0": t0, "image1": t1}
        correspondences = self.matcher(input_dict)

        kpts0 = correspondences['keypoints0'].cpu().numpy()
        kpts1 = correspondences['keypoints1'].cpu().numpy()
        conf = correspondences['confidence'].cpu().numpy()

        # Filter by confidence threshold
        mask = conf >= conf_thresh
        kpts0 = kpts0[mask]
        kpts1 = kpts1[mask]
        conf = conf[mask]

        return {
            "mkpts0": kpts0,
            "mkpts1": kpts1,
            "conf": conf
        }


class ScaleSpacePyramidMatcher:
    """
    Multi-Scale Feature Matching Engine.
    Enables correspondence across heterogeneous sensors with severe resolution discrepancy
    (e.g., OHRC ~0.25m/px vs TMC ~5m/px vs IIRS ~80m/px).
    """

    def __init__(self, base_matcher: Optional[LunarLoFTRMatcher] = None, device: Optional[str] = None):
        self.matcher = base_matcher if base_matcher is not None else LunarLoFTRMatcher(device=device)

    def match_multiscale(
        self,
        img0_path: str,
        img1_path: str,
        scale_pyramid: List[float] = [1.0, 0.5, 0.25],
        conf_thresh: float = 0.2,
        resize_long: int = 1024
    ) -> Dict[str, Any]:
        """
        Tests multi-scale octaves to discover optimal cross-sensor resolution alignment.
        """
        img0_scaled, img1_scaled, s0, s1 = preprocess_lunar_pair(
            img0_path, img1_path, resize_long=resize_long, use_multiscale=True
        )

        best_result = None
        best_inliers = -1
        best_scale_factor = 1.0

        for pyr_scale in scale_pyramid:
            if pyr_scale == 1.0:
                cur_img1 = img1_scaled
                cur_s1 = s1
            else:
                nh = int(round(img1_scaled.shape[0] * pyr_scale / 8.0)) * 8
                nw = int(round(img1_scaled.shape[1] * pyr_scale / 8.0)) * 8
                if nh < 64 or nw < 64:
                    continue
                cur_img1 = cv2.resize(img1_scaled, (nw, nh), interpolation=cv2.INTER_AREA)
                cur_s1 = s1 * pyr_scale

            res = self.matcher.match(img0_scaled, cur_img1, conf_thresh=conf_thresh)
            num_pts = len(res['mkpts0'])

            if num_pts > best_inliers:
                best_inliers = num_pts
                best_scale_factor = pyr_scale
                # Map points back to original image coordinate spaces
                mkpts0_orig = res['mkpts0'] / s0
                mkpts1_orig = res['mkpts1'] / cur_s1

                best_result = {
                    "mkpts0_scaled": res['mkpts0'],
                    "mkpts1_scaled": res['mkpts1'],
                    "mkpts0_orig": mkpts0_orig,
                    "mkpts1_orig": mkpts1_orig,
                    "conf": res['conf'],
                    "best_scale_factor": best_scale_factor,
                    "scale0": s0,
                    "scale1": cur_s1,
                    "img0_scaled": img0_scaled,
                    "img1_scaled": cur_img1
                }

        if best_result is None:
            # Fallback empty result
            best_result = {
                "mkpts0_scaled": np.empty((0, 2), dtype=np.float32),
                "mkpts1_scaled": np.empty((0, 2), dtype=np.float32),
                "mkpts0_orig": np.empty((0, 2), dtype=np.float32),
                "mkpts1_orig": np.empty((0, 2), dtype=np.float32),
                "conf": np.empty((0,), dtype=np.float32),
                "best_scale_factor": 1.0,
                "scale0": s0,
                "scale1": s1,
                "img0_scaled": img0_scaled,
                "img1_scaled": img1_scaled
            }

        return best_result
