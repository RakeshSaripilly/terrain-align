"""
SunAngle/matching_engine.py
Deep Transformer Feature Extraction and Scale-Space Matching Engine
Multi-modal and scale-invariant correspondence engine for Chandrayaan-2 (OHRC, TMC, IIRS).
"""

from typing import Dict, Any, Tuple, Optional, List, Union
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


def generate_adaptive_scale_pyramid(scale_ratio: float) -> List[float]:
    """
    Generates adaptive multi-scale pyramid octaves based on physical GSD ratio.
    Ensures LoFTR evaluates feature correspondences across octaves for cross-resolution pairs.
    """
    ratio = float(scale_ratio) if scale_ratio > 0 else 1.0
    r = max(ratio, 1.0 / ratio)

    if r < 1.3:
        # Same-scale or same-sensor pairs (e.g. NAC-NAC): base scale is sufficient
        return [1.0]
    elif r <= 4.0:
        # Moderate scale gap (e.g. 2x - 4x)
        return [1.0, 0.5]
    elif r <= 16.0:
        # Medium-large scale gap (e.g. TMC-IIRS ~16x)
        return [1.0, 0.5, 0.25]
    else:
        # Large/extreme scale gap (e.g. OHRC-TMC ~20x, OHRC-IIRS ~320x)
        return [1.0, 0.5, 0.25, 0.125]


def deduplicate_match_points(
    pts0: np.ndarray,
    pts1: np.ndarray,
    conf: np.ndarray,
    dist_thresh: float = 3.0
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Deduplicates merged multi-scale keypoint correspondences that fall within dist_thresh pixels.
    Keeps the correspondence with highest confidence.
    """
    if len(pts0) <= 1:
        return pts0, pts1, conf

    # Sort descending by confidence
    order = np.argsort(-conf)
    s_pts0 = pts0[order]
    s_pts1 = pts1[order]
    s_conf = conf[order]

    keep = []
    # Grid-based spatial hashing for O(N) neighbor lookup
    cell_size = max(dist_thresh, 1.0)
    grid: Dict[Tuple[int, int], List[int]] = {}

    for i in range(len(s_pts0)):
        gx = int(s_pts0[i, 0] / cell_size)
        gy = int(s_pts0[i, 1] / cell_size)
        is_dup = False

        # Check 3x3 neighboring cells
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                neighbor_key = (gx + dx, gy + dy)
                if neighbor_key in grid:
                    for prev_idx in grid[neighbor_key]:
                        d0 = np.linalg.norm(s_pts0[i] - s_pts0[prev_idx])
                        d1 = np.linalg.norm(s_pts1[i] - s_pts1[prev_idx])
                        if d0 < dist_thresh and d1 < dist_thresh:
                            is_dup = True
                            break
                if is_dup:
                    break
            if is_dup:
                break

        if not is_dup:
            key = (gx, gy)
            if key not in grid:
                grid[key] = []
            grid[key].append(i)
            keep.append(i)

    keep_idx = np.array(keep, dtype=int)
    return s_pts0[keep_idx], s_pts1[keep_idx], s_conf[keep_idx]


class ScaleSpacePyramidMatcher:
    """
    Multi-Scale Feature Matching Engine.
    Enables correspondence across heterogeneous sensors with severe resolution discrepancy
    (e.g., OHRC ~0.25m/px vs TMC ~5m/px vs IIRS ~80m/px).
    
    Key Features:
    - Adaptive pyramid levels based on physical GSD ratio.
    - Geometric quality selection (USAC_MAGSAC inliers & RMSE, not just raw match count).
    - Multi-level feature merging with spatial deduplication.
    """

    def __init__(self, base_matcher: Optional[LunarLoFTRMatcher] = None, device: Optional[str] = None):
        self.matcher = base_matcher if base_matcher is not None else LunarLoFTRMatcher(device=device)

    def match_scaled_pair(
        self,
        img0_scaled: np.ndarray,
        img1_scaled: np.ndarray,
        scale_ratio: float = 1.0,
        scale_pyramid: Optional[List[float]] = None,
        conf_thresh: float = 0.20,
        ransac_thresh: float = 2.0,
        merge_levels: bool = True
    ) -> Dict[str, Any]:
        """
        Executes multi-scale pyramid matching on pre-scaled / GSD-normalized image pairs.
        Evaluates each level using MAGSAC++ geometric quality.
        """
        if scale_pyramid is None:
            scale_pyramid = generate_adaptive_scale_pyramid(scale_ratio)

        h0, w0 = img0_scaled.shape[:2]
        h1, w1 = img1_scaled.shape[:2]

        level_results = []
        best_level = None
        best_score = -1.0

        for pyr_scale in scale_pyramid:
            if pyr_scale == 1.0:
                p_img0 = img0_scaled
                p_img1 = img1_scaled
                scale0_lvl, scale1_lvl = 1.0, 1.0
            else:
                nw0 = max((int(round(w0 * pyr_scale)) // 8) * 8, 64)
                nh0 = max((int(round(h0 * pyr_scale)) // 8) * 8, 64)
                nw1 = max((int(round(w1 * pyr_scale)) // 8) * 8, 64)
                nh1 = max((int(round(h1 * pyr_scale)) // 8) * 8, 64)

                if min(nw0, nh0, nw1, nh1) < 64:
                    continue

                p_img0 = cv2.resize(img0_scaled, (nw0, nh0), interpolation=cv2.INTER_AREA)
                p_img1 = cv2.resize(img1_scaled, (nw1, nh1), interpolation=cv2.INTER_AREA)
                scale0_lvl = nw0 / float(w0)
                scale1_lvl = nw1 / float(w1)

            # Match at this pyramid level
            res = self.matcher.match(p_img0, p_img1, conf_thresh=conf_thresh)
            kpts0 = res['mkpts0']
            kpts1 = res['mkpts1']
            conf = res['conf']

            if len(kpts0) == 0:
                continue

            # Map points back to base matcher coordinates (img0_scaled and img1_scaled coordinate frame)
            kpts0_base = kpts0 / scale0_lvl
            kpts1_base = kpts1 / scale1_lvl

            # Evaluate geometric quality using MAGSAC++
            inlier_count = 0
            inlier_ratio = 0.0
            rmse = float('inf')
            inlier_mask = np.zeros(len(kpts0_base), dtype=bool)
            H = None

            if len(kpts0_base) >= 4:
                try:
                    H_est, mask_est = cv2.findHomography(
                        kpts0_base, kpts1_base,
                        method=cv2.USAC_MAGSAC,
                        ransacReprojThreshold=ransac_thresh,
                        maxIters=5000,
                        confidence=0.995
                    )
                    if H_est is not None and mask_est is not None:
                        H = H_est
                        inlier_mask = mask_est.ravel().astype(bool)
                        inlier_count = int(np.sum(inlier_mask))
                        inlier_ratio = inlier_count / float(len(kpts0_base))
                        if inlier_count > 0:
                            p0_in = kpts0_base[inlier_mask]
                            p1_in = kpts1_base[inlier_mask]
                            p0_h = np.column_stack([p0_in, np.ones(len(p0_in))])
                            pred1_h = (H @ p0_h.T).T
                            pred1 = pred1_h[:, :2] / (pred1_h[:, 2:3] + 1e-12)
                            err = np.linalg.norm(p1_in - pred1, axis=1)
                            rmse = float(np.sqrt(np.mean(err ** 2)))
                except Exception:
                    pass

            # Geometric selection score: prioritizes confident inlier counts with low reprojection error
            geom_score = inlier_count * (inlier_ratio ** 1.2) / (rmse + 0.2) if inlier_count >= 4 else float(len(kpts0)) * 0.01

            level_data = {
                "scale": pyr_scale,
                "raw_matches": len(kpts0),
                "inlier_count": inlier_count,
                "inlier_ratio": inlier_ratio,
                "rmse": rmse,
                "geom_score": geom_score,
                "pts0_base": kpts0_base,
                "pts1_base": kpts1_base,
                "conf": conf,
                "inlier_mask": inlier_mask,
                "H": H
            }
            level_results.append(level_data)

            if geom_score > best_score:
                best_score = geom_score
                best_level = level_data

        if best_level is None:
            # Fallback empty result
            return {
                "mkpts0": np.empty((0, 2), dtype=np.float32),
                "mkpts1": np.empty((0, 2), dtype=np.float32),
                "conf": np.empty((0,), dtype=np.float32),
                "inlier_mask": np.empty((0,), dtype=bool),
                "inlier_count": 0,
                "inlier_ratio": 0.0,
                "rmse": float('inf'),
                "H": None,
                "best_scale_factor": 1.0,
                "level_results": level_results
            }

        # Multi-level merging: combine geometrically consistent points from useful levels
        if merge_levels and len(level_results) > 1:
            useful_levels = [lvl for lvl in level_results if lvl["inlier_count"] >= 4 and lvl["inlier_ratio"] >= 0.15]
            if len(useful_levels) > 1:
                all_p0 = np.vstack([lvl["pts0_base"][lvl["inlier_mask"]] for lvl in useful_levels])
                all_p1 = np.vstack([lvl["pts1_base"][lvl["inlier_mask"]] for lvl in useful_levels])
                all_conf = np.concatenate([lvl["conf"][lvl["inlier_mask"]] for lvl in useful_levels])

                # Deduplicate points that fell within 3 pixels across octave scales
                dedup_p0, dedup_p1, dedup_conf = deduplicate_match_points(all_p0, all_p1, all_conf, dist_thresh=3.0)

                if len(dedup_p0) >= 4:
                    try:
                        H_merged, mask_merged = cv2.findHomography(
                            dedup_p0, dedup_p1,
                            method=cv2.USAC_MAGSAC,
                            ransacReprojThreshold=ransac_thresh,
                            maxIters=5000,
                            confidence=0.995
                        )
                        if H_merged is not None and mask_merged is not None:
                            inl_mask_m = mask_merged.ravel().astype(bool)
                            m_count = int(np.sum(inl_mask_m))
                            if m_count >= best_level["inlier_count"]:
                                p0_in = dedup_p0[inl_mask_m]
                                p1_in = dedup_p1[inl_mask_m]
                                p0_h = np.column_stack([p0_in, np.ones(len(p0_in))])
                                pred1_h = (H_merged @ p0_h.T).T
                                pred1 = pred1_h[:, :2] / (pred1_h[:, 2:3] + 1e-12)
                                err = np.linalg.norm(p1_in - pred1, axis=1)
                                m_rmse = float(np.sqrt(np.mean(err ** 2)))

                                if m_rmse <= best_level["rmse"] * 1.3:
                                    return {
                                        "mkpts0": dedup_p0,
                                        "mkpts1": dedup_p1,
                                        "conf": dedup_conf,
                                        "inlier_mask": inl_mask_m,
                                        "inlier_count": m_count,
                                        "inlier_ratio": m_count / float(len(dedup_p0)),
                                        "rmse": m_rmse,
                                        "H": H_merged,
                                        "best_scale_factor": best_level["scale"],
                                        "level_results": level_results,
                                        "merged": True
                                    }
                    except Exception:
                        pass

        # Return single best pyramid level
        return {
            "mkpts0": best_level["pts0_base"],
            "mkpts1": best_level["pts1_base"],
            "conf": best_level["conf"],
            "inlier_mask": best_level["inlier_mask"],
            "inlier_count": best_level["inlier_count"],
            "inlier_ratio": best_level["inlier_ratio"],
            "rmse": best_level["rmse"],
            "H": best_level["H"],
            "best_scale_factor": best_level["scale"],
            "level_results": level_results,
            "merged": False
        }

    def match_multiscale(
        self,
        img0_path: Union[str, np.ndarray],
        img1_path: Union[str, np.ndarray],
        scale_pyramid: Optional[List[float]] = None,
        conf_thresh: float = 0.2,
        resize_long: int = 1024,
        gsd0: Optional[float] = None,
        gsd1: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Loads pair, normalizes GSD, and tests multi-scale octaves.
        """
        img0_scaled, img1_scaled, s0, s1 = preprocess_lunar_pair(
            img0_path, img1_path, resize_long=resize_long, use_multiscale=True
        )
        scale_ratio = float(gsd0 / gsd1) if (gsd0 and gsd1) else 1.0
        pyr_res = self.match_scaled_pair(
            img0_scaled, img1_scaled,
            scale_ratio=scale_ratio,
            scale_pyramid=scale_pyramid,
            conf_thresh=conf_thresh
        )

        mkpts0_scaled = pyr_res["mkpts0"]
        mkpts1_scaled = pyr_res["mkpts1"]
        mkpts0_orig = mkpts0_scaled / s0 if len(mkpts0_scaled) > 0 else np.empty((0, 2), dtype=np.float32)
        mkpts1_orig = mkpts1_scaled / s1 if len(mkpts1_scaled) > 0 else np.empty((0, 2), dtype=np.float32)

        return {
            "mkpts0_scaled": mkpts0_scaled,
            "mkpts1_scaled": mkpts1_scaled,
            "mkpts0_orig": mkpts0_orig,
            "mkpts1_orig": mkpts1_orig,
            "conf": pyr_res["conf"],
            "inlier_mask": pyr_res["inlier_mask"],
            "inlier_count": pyr_res["inlier_count"],
            "inlier_ratio": pyr_res["inlier_ratio"],
            "rmse": pyr_res["rmse"],
            "H": pyr_res["H"],
            "best_scale_factor": pyr_res["best_scale_factor"],
            "scale0": s0,
            "scale1": s1,
            "img0_scaled": img0_scaled,
            "img1_scaled": img1_scaled,
            "level_results": pyr_res.get("level_results", [])
        }
