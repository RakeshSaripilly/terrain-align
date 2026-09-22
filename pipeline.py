"""
SunAngle/pipeline.py
Unified High-Level Surface Feature Matching and Image Correspondence Pipeline
Robust multi-modal and illumination-invariant surface image correspondence.
"""

import os
import time
from typing import Dict, Any, Optional, Union
import cv2
import numpy as np

from SunAngle.preprocessing import (
    load_lunar_image,
    preprocess_lunar_pair,
    local_contrast_normalization,
    multi_scale_lcn
)
from SunAngle.matching_engine import LunarLoFTRMatcher
from SunAngle.subpixel_uniformity import (
    refine_subpixel_lk,
    enforce_spatial_uniformity_anms,
    quadtree_spatial_binning,
    compute_spatial_uniformity_metrics
)
from SunAngle.geometry_warping import (
    estimate_robust_transformation,
    warp_lunar_image,
    generate_checkerboard,
    compute_difference_map
)
from SunAngle.metrics_and_vis import (
    compute_alignment_similarity_metrics,
    visualize_matches,
    visualize_spatial_distribution
)


class LunarCorrespondenceEngine:
    """
    End-to-end Computer Vision & Remote Sensing Engine for Chandrayaan-2 lunar imagery.
    Delivers sub-pixel registration accuracy (target RMSE < 0.5 pixels) across
    extreme sun angle changes (30° - 85°) and multi-modal sensor resolutions.
    """

    def __init__(self, device: Optional[str] = None):
        self.matcher = LunarLoFTRMatcher(pretrained='outdoor', device=device)

    def match(
        self,
        img0_input: Union[str, np.ndarray],
        img1_input: Union[str, np.ndarray],
        resize_long: int = 1024,
        conf_thresh: float = 0.20,
        subpixel_refinement: bool = True,
        enforce_uniformity: bool = True,
        max_uniform_points: int = 1200,
        ransac_thresh: float = 2.0
    ) -> Dict[str, Any]:
        """
        Executes full correspondence pipeline:
        MS-LCN -> LoFTR -> Uniform Sampling -> Subpixel LK -> USAC_MAGSAC.
        """
        start_time = time.perf_counter()

        # 1. Radiometric & Multi-Scale LCN Preprocessing
        img0_scaled, img1_scaled, s0, s1 = preprocess_lunar_pair(
            img0_input, img1_input, resize_long=resize_long, use_multiscale=True
        )

        # 2. Deep Transformer Matching
        match_start = time.perf_counter()
        raw_res = self.matcher.match(img0_scaled, img1_scaled, conf_thresh=conf_thresh)
        raw_pts0 = raw_res['mkpts0']
        raw_pts1 = raw_res['mkpts1']
        raw_conf = raw_res['conf']
        match_time = time.perf_counter() - match_start

        # 3. Spatial Uniformity Enforcement (prevent crater rim over-clustering)
        if enforce_uniformity and len(raw_pts0) > 100:
            unif_pts0, unif_pts1, unif_conf = quadtree_spatial_binning(
                raw_pts0, raw_pts1, raw_conf,
                img_shape=img0_scaled.shape,
                grid_size=8,
                points_per_bin=max_uniform_points // 64 + 10
            )
        else:
            unif_pts0, unif_pts1, unif_conf = raw_pts0, raw_pts1, raw_conf

        # 4. Sub-Pixel Precision Refinement (Inverse-Compositional Lucas-Kanade)
        if subpixel_refinement and len(unif_pts0) >= 4:
            ref_pts0, ref_pts1, lk_mask = refine_subpixel_lk(
                img0_scaled, img1_scaled, unif_pts0, unif_pts1, window_size=15
            )
            pts0_to_estimate = ref_pts0
            pts1_to_estimate = ref_pts1
            conf_to_estimate = unif_conf[lk_mask] if len(unif_conf) == len(lk_mask) else np.ones(len(ref_pts0))
        else:
            pts0_to_estimate = unif_pts0
            pts1_to_estimate = unif_pts1
            conf_to_estimate = unif_conf

        # 5. Robust Geometric Transformation & Outlier Rejection (USAC_MAGSAC)
        geom = estimate_robust_transformation(
            pts0_to_estimate, pts1_to_estimate, ransac_thresh=ransac_thresh
        )

        # 6. Spatial Uniformity Metrics
        uniformity_metrics = compute_spatial_uniformity_metrics(
            geom["inlier_pts0"] if geom["success"] else pts0_to_estimate,
            img_shape=img0_scaled.shape,
            grid_divisions=6
        )

        total_time = time.perf_counter() - start_time

        # Convert coordinates to original image space
        orig_inliers0 = geom["inlier_pts0"] / s0 if geom["success"] else np.array([])
        orig_inliers1 = geom["inlier_pts1"] / s1 if geom["success"] else np.array([])

        return {
            "success": geom["success"],
            "raw_matches": len(raw_pts0),
            "uniform_matches": len(unif_pts0),
            "inlier_count": geom["inlier_count"],
            "inlier_ratio": geom["inlier_ratio"],
            "rmse_pixels": geom["rmse"],
            "subpixel_accurate": bool(geom["rmse"] < 0.5),
            "spatial_coverage": uniformity_metrics["coverage_fraction"],
            "spatial_entropy": uniformity_metrics["spatial_entropy"],
            "H_scaled": geom["H"].tolist() if geom["H"] is not None else None,
            "scale0": s0,
            "scale1": s1,
            "pts0_scaled": pts0_to_estimate,
            "pts1_scaled": pts1_to_estimate,
            "inlier_mask": geom["inlier_mask"],
            "orig_inliers0": orig_inliers0,
            "orig_inliers1": orig_inliers1,
            "img0_scaled": img0_scaled,
            "img1_scaled": img1_scaled,
            "elapsed_total_sec": round(total_time, 3),
            "elapsed_match_sec": round(match_time, 3)
        }

    def register_and_visualize(
        self,
        img0_path: str,
        img1_path: str,
        out_dir: str = "outputs",
        resize_long: int = 1024
    ) -> Dict[str, Any]:
        """
        Runs complete matching, warps moving image, and writes all visual artifacts.
        """
        os.makedirs(out_dir, exist_ok=True)

        res = self.match(
            img0_path, img1_path, resize_long=resize_long, subpixel_refinement=True
        )

        img0_scaled = res["img0_scaled"]
        img1_scaled = res["img1_scaled"]
        H = np.array(res["H_scaled"]) if res["H_scaled"] is not None else None

        # 1. Match Lines Visualization
        match_vis_path = os.path.join(out_dir, "match_lines.jpg")
        visualize_matches(
            img0_scaled, img1_scaled,
            res["pts0_scaled"], res["pts1_scaled"],
            res["inlier_mask"],
            out_path=match_vis_path
        )

        # 2. Spatial Uniformity Grid
        density_path = os.path.join(out_dir, "spatial_density.jpg")
        inlier_pts0 = res["pts0_scaled"][res["inlier_mask"]] if res["success"] else res["pts0_scaled"]
        visualize_spatial_distribution(img0_scaled.shape, inlier_pts0, out_path=density_path)

        similarity_metrics = {"ncc": 0.0, "mutual_information": 0.0, "psnr": 0.0}
        if res["success"] and H is not None:
            # Warp img1 onto img0 frame (img1 -> img0 requires H_inv if H is img0 -> img1)
            H_inv = np.linalg.inv(H)
            img1_warped = warp_lunar_image(img1_scaled, img0_scaled.shape, H_inv)

            # 3. Checkerboard Overlay
            checker_path = os.path.join(out_dir, "checkerboard_overlay.jpg")
            checkerboard = generate_checkerboard(img0_scaled, img1_warped, square_size=64)
            cv2.imwrite(checker_path, checkerboard)

            # 4. Difference Heatmap
            diff_path = os.path.join(out_dir, "difference_heatmap.jpg")
            _, heatmap = compute_difference_map(img0_scaled, img1_warped)
            cv2.imwrite(diff_path, heatmap)

            # 5. Save Warped Registered Image
            warped_path = os.path.join(out_dir, "registered_target.jpg")
            cv2.imwrite(warped_path, img1_warped)

            similarity_metrics = compute_alignment_similarity_metrics(img0_scaled, img1_warped)

        # 6. Save normalized LCN images for presentation
        cv2.imwrite(os.path.join(out_dir, "lcn_ref.jpg"), img0_scaled)
        cv2.imwrite(os.path.join(out_dir, "lcn_target.jpg"), img1_scaled)

        # Combine all metrics
        output_metrics = {
            "success": res["success"],
            "raw_matches": res["raw_matches"],
            "uniform_matches": res["uniform_matches"],
            "inlier_count": res["inlier_count"],
            "inlier_ratio": res["inlier_ratio"],
            "rmse_pixels": res["rmse_pixels"],
            "subpixel_accuracy_achieved": res["subpixel_accurate"],
            "spatial_coverage": res["spatial_coverage"],
            "spatial_entropy": res["spatial_entropy"],
            "ncc_cross_correlation": similarity_metrics["ncc"],
            "mutual_information": similarity_metrics["mutual_information"],
            "psnr_db": similarity_metrics["psnr"],
            "match_time_sec": res["elapsed_match_sec"],
            "total_time_sec": res["elapsed_total_sec"]
        }

        return output_metrics
