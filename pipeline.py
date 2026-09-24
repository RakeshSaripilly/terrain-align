import os
import sys
import time
from pathlib import Path
from typing import Dict, Any, Optional, Union, Tuple
import cv2
import numpy as np

# Try relative/package imports, fallback to local/flat imports
try:
    from .preprocessing import (
        load_lunar_image,
        preprocess_lunar_pair,
        create_gsd_normalized_pair,
        apply_coordinate_transform,
        infer_gsd_from_path,
        local_contrast_normalization,
        multi_scale_lcn,
        SENSOR_GSD
    )
    from .matching_engine import LunarLoFTRMatcher, ScaleSpacePyramidMatcher, generate_adaptive_scale_pyramid
    from .subpixel_uniformity import (
        refine_subpixel_lk,
        refine_subpixel_lk_native,
        refine_subpixel_gruen_lsm,
        refine_subpixel_hybrid,
        enforce_spatial_uniformity_anms,
        quadtree_spatial_binning,
        spatial_grid_quota_binning,
        compute_spatial_uniformity_metrics,
        compute_sdi_metric
    )
    from .geometry_warping import (
        estimate_robust_transformation,
        warp_lunar_image,
        warp_lunar_image_memory_safe,
        generate_checkerboard,
        compute_difference_map,
        compute_scale_consistency
    )
    from .evaluation import compute_checkpoint_rmse
    from .metrics_and_vis import (
        compute_alignment_similarity_metrics,
        visualize_matches,
        visualize_spatial_distribution
    )
except (ImportError, ValueError):
    from preprocessing import (
        load_lunar_image,
        preprocess_lunar_pair,
        create_gsd_normalized_pair,
        apply_coordinate_transform,
        infer_gsd_from_path,
        local_contrast_normalization,
        multi_scale_lcn,
        SENSOR_GSD
    )
    from matching_engine import LunarLoFTRMatcher, ScaleSpacePyramidMatcher, generate_adaptive_scale_pyramid
    from subpixel_uniformity import (
        refine_subpixel_lk,
        refine_subpixel_lk_native,
        refine_subpixel_gruen_lsm,
        refine_subpixel_hybrid,
        enforce_spatial_uniformity_anms,
        quadtree_spatial_binning,
        spatial_grid_quota_binning,
        compute_spatial_uniformity_metrics,
        compute_sdi_metric
    )
    from geometry_warping import (
        estimate_robust_transformation,
        warp_lunar_image,
        warp_lunar_image_memory_safe,
        generate_checkerboard,
        compute_difference_map,
        compute_scale_consistency
    )
    from evaluation import compute_checkpoint_rmse
    from metrics_and_vis import (
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
        self.device = device
        self.matcher = LunarLoFTRMatcher(pretrained='outdoor', device=device)
        self.pyramid_matcher = ScaleSpacePyramidMatcher(base_matcher=self.matcher, device=device)

    def match(
        self,
        img0_input: Union[str, np.ndarray],
        img1_input: Union[str, np.ndarray],
        resize_long: int = 1024,
        conf_thresh: float = 0.20,
        subpixel_refinement: bool = True,
        enforce_uniformity: bool = True,
        max_uniform_points: int = 1200,
        ransac_thresh: float = 2.0,
        gsd0: Optional[float] = None,
        gsd1: Optional[float] = None,
        sensor0: Optional[str] = None,
        sensor1: Optional[str] = None,
        k_min: int = 5,
        k_max: int = 25,
        model: str = "homography"
    ) -> Dict[str, Any]:
        """
        Executes GSD-normalized hierarchical correspondence pipeline:
        Physical GSD Normalization -> Multi-Scale LoFTR -> MAGSAC++ Verification ->
        Spatial Grid Quota Binning -> Sub-Pixel Gruen LSM / Native LK -> Final Model Fit.
        """
        start_time = time.perf_counter()

        # Resolve GSDs if sensor names or paths provided
        if gsd0 is None and sensor0:
            gsd0 = SENSOR_GSD.get(sensor0.lower(), None)
        if gsd1 is None and sensor1:
            gsd1 = SENSOR_GSD.get(sensor1.lower(), None)

        if gsd0 is None and isinstance(img0_input, (str, Path)):
            gsd0 = infer_gsd_from_path(img0_input)
        if gsd1 is None and isinstance(img1_input, (str, Path)):
            gsd1 = infer_gsd_from_path(img1_input)

        # 1. GSD Scale Normalization Layer (MTF anti-aliasing & common physical GSD)
        gsd_prep = create_gsd_normalized_pair(
            img_src_input=img1_input,
            img_ref_input=img0_input,
            gsd_src=gsd1,
            gsd_ref=gsd0,
            resize_long=resize_long,
            use_multiscale=True,
            verbose=True
        )

        img0_scaled = gsd_prep["img_ref_scaled"]
        img1_scaled = gsd_prep["img_src_scaled"]
        img0_native = gsd_prep["img_ref_native"]
        img1_native = gsd_prep["img_src_native"]
        s0_total = gsd_prep["scale_ref_total"]
        s1_total = gsd_prep["scale_src_total"]
        scale_ratio = gsd_prep["scale_ratio"]
        common_gsd = gsd_prep["common_gsd"]
        gsd_ref_final = gsd_prep["gsd_ref"]
        gsd_src_final = gsd_prep["gsd_src"]

        T_ref_nat_to_match = gsd_prep["T_ref_native_to_matcher"]
        T_ref_match_to_nat = gsd_prep["T_ref_matcher_to_native"]
        T_src_nat_to_match = gsd_prep["T_src_native_to_matcher"]
        T_src_match_to_nat = gsd_prep["T_src_matcher_to_native"]

        # 2. Scale-Aware Deep Correspondence Matching via ScaleSpacePyramidMatcher
        match_start = time.perf_counter()
        pyr_res = self.pyramid_matcher.match_scaled_pair(
            img0_scaled=img0_scaled,
            img1_scaled=img1_scaled,
            scale_ratio=scale_ratio,
            conf_thresh=conf_thresh,
            ransac_thresh=ransac_thresh,
            merge_levels=True
        )

        raw_pts0 = pyr_res["mkpts0"]
        raw_pts1 = pyr_res["mkpts1"]
        raw_conf = pyr_res["conf"]
        match_time = time.perf_counter() - match_start

        # 3. Geometric Verification (MAGSAC++) BEFORE Spatial Quota Filtering
        # Spatial quota filtering should not determine correspondence quality before geometric verification.
        # Geometry establishes which candidate matches are actually consistent.
        initial_geom = estimate_robust_transformation(
            raw_pts0, raw_pts1,
            ransac_thresh=ransac_thresh,
            model=model,
            gsd_ratio=scale_ratio,
            expected_residual_scale=1.0
        )

        if initial_geom["success"] and initial_geom["inlier_count"] >= 4:
            geom_inlier_mask = initial_geom["inlier_mask"]
            candidate_pts0 = raw_pts0[geom_inlier_mask]
            candidate_pts1 = raw_pts1[geom_inlier_mask]
            candidate_conf = raw_conf[geom_inlier_mask]
        else:
            candidate_pts0 = raw_pts0
            candidate_pts1 = raw_pts1
            candidate_conf = raw_conf

        # 4. Spatial Grid Partitioning (Uniform Distribution Enforcement on Verified Inliers)
        if enforce_uniformity and len(candidate_pts0) > 10:
            unif_pts0, unif_pts1, unif_conf = spatial_grid_quota_binning(
                candidate_pts0, candidate_pts1, candidate_conf,
                img_shape=img0_scaled.shape,
                grid_size=8,
                k_min=k_min,
                k_max=k_max
            )
            # ANMS if count exceeds max_uniform_points
            if len(unif_pts0) > max_uniform_points:
                unif_pts0, unif_pts1, unif_conf = enforce_spatial_uniformity_anms(
                    unif_pts0, unif_pts1, unif_conf, max_points=max_uniform_points
                )
        else:
            unif_pts0, unif_pts1, unif_conf = candidate_pts0, candidate_pts1, candidate_conf

        # 5. Sub-Pixel Back-Projection & Local Refinement in Native Resolution
        # Use exact affine coordinate transforms instead of scalar approximations:
        pts0_native = apply_coordinate_transform(unif_pts0, T_ref_match_to_nat)
        pts1_native = apply_coordinate_transform(unif_pts1, T_src_match_to_nat)

        if subpixel_refinement and len(pts0_native) >= 4:
            # Hybrid refinement: Gruen Least Squares Matching (LSM) + cornerSubPix/LK fallback
            ref_pts0_nat, ref_pts1_nat, sub_mask = refine_subpixel_hybrid(
                img0_native, img1_native, pts0_native, pts1_native,
                window_size=15, max_iters=25
            )
            pts0_to_estimate = apply_coordinate_transform(ref_pts0_nat, T_ref_nat_to_match)
            pts1_to_estimate = apply_coordinate_transform(ref_pts1_nat, T_src_nat_to_match)
            conf_to_estimate = unif_conf[sub_mask] if len(unif_conf) == len(sub_mask) else np.ones(len(pts0_to_estimate), dtype=np.float32)
        else:
            pts0_to_estimate = unif_pts0
            pts1_to_estimate = unif_pts1
            conf_to_estimate = unif_conf

        # 6. Final Robust Geometric Transformation & Outlier Rejection
        geom = estimate_robust_transformation(
            pts0_to_estimate, pts1_to_estimate,
            ransac_thresh=ransac_thresh,
            model=model,
            gsd_ratio=scale_ratio,
            expected_residual_scale=1.0
        )

        inlier_pts0 = geom["inlier_pts0"] if geom["success"] else pts0_to_estimate
        inlier_pts1 = geom["inlier_pts1"] if geom["success"] else pts1_to_estimate

        # 7. Spatial Distribution Index (SDI) and ISRO Uniformity
        uniformity_metrics = compute_spatial_uniformity_metrics(
            inlier_pts0,
            img_shape=img0_scaled.shape,
            grid_divisions=8
        )
        sdi_val = compute_sdi_metric(inlier_pts0, img_shape=img0_scaled.shape, grid_divisions=8)

        # 8. 80/20 Tie-Point / Check-Point Reprojection RMSE
        chk_metrics = compute_checkpoint_rmse(
            inlier_pts0, inlier_pts1, geom["H"],
            split_ratio=0.8, gsd_ref=gsd_ref_final
        )

        total_time = time.perf_counter() - start_time

        # Convert inliers to native full-resolution coordinates via exact coordinate transforms
        orig_inliers0 = apply_coordinate_transform(inlier_pts0, T_ref_match_to_nat) if geom["success"] else np.empty((0, 2), dtype=np.float32)
        orig_inliers1 = apply_coordinate_transform(inlier_pts1, T_src_match_to_nat) if geom["success"] else np.empty((0, 2), dtype=np.float32)

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
            "sdi": sdi_val,
            "sdi_8x8": sdi_val,
            "checkpoint_rmse_px": chk_metrics["checkpoint_rmse_px"],
            "checkpoint_rmse_meters": chk_metrics["checkpoint_rmse_meters"],
            "tiepoint_rmse_px": chk_metrics["tiepoint_rmse_px"],
            "tiepoint_rmse_meters": chk_metrics["tiepoint_rmse_meters"],
            "scale_consistency": geom.get("scale_consistency", {}),
            "scale_ratio": scale_ratio,
            "common_gsd": common_gsd,
            "gsd_ref": gsd_ref_final,
            "gsd_src": gsd_src_final,
            "H_scaled": geom["H"].tolist() if geom["H"] is not None else None,
            "H_matrix": geom["H"],
            "tps": geom.get("tps"),
            "scale0": s0_total,
            "scale1": s1_total,
            "pts0_scaled": pts0_to_estimate,
            "pts1_scaled": pts1_to_estimate,
            "inlier_mask": geom["inlier_mask"],
            "residuals": geom["residuals"],
            "orig_inliers0": orig_inliers0,
            "orig_inliers1": orig_inliers1,
            "img0_scaled": img0_scaled,
            "img1_scaled": img1_scaled,
            "img0_native": img0_native,
            "img1_native": img1_native,
            "T_ref_native_to_matcher": T_ref_nat_to_match,
            "T_ref_matcher_to_native": T_ref_match_to_nat,
            "T_src_native_to_matcher": T_src_nat_to_match,
            "T_src_matcher_to_native": T_src_match_to_nat,
            "elapsed_total_sec": round(total_time, 3),
            "elapsed_match_sec": round(match_time, 3)
        }

    def register_and_visualize(
        self,
        img0_path: str,
        img1_path: str,
        out_dir: str = "outputs",
        resize_long: int = 1024,
        sensor0: Optional[str] = None,
        sensor1: Optional[str] = None,
        gsd0: Optional[float] = None,
        gsd1: Optional[float] = None,
        model: str = "homography"
    ) -> Dict[str, Any]:
        """
        Runs complete matching, warps moving image, and writes all visual artifacts.
        """
        os.makedirs(out_dir, exist_ok=True)

        res = self.match(
            img0_path, img1_path,
            resize_long=resize_long,
            subpixel_refinement=True,
            sensor0=sensor0, sensor1=sensor1,
            gsd0=gsd0, gsd1=gsd1,
            model=model
        )

        img0_scaled = res["img0_scaled"]
        img1_scaled = res["img1_scaled"]
        H = res["H_matrix"]

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
        if res["success"]:
            try:
                # Warp img1 onto img0 frame
                if res.get("tps") is not None and model.lower() == "tps":
                    img1_warped = warp_lunar_image(img1_scaled, img0_scaled.shape, res["tps"])
                elif H is not None:
                    H_inv = np.linalg.inv(H)
                    img1_warped = warp_lunar_image(img1_scaled, img0_scaled.shape, H_inv)
                else:
                    img1_warped = img1_scaled

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
            except Exception as e:
                print(f"[WARP-WARNING] Visual warping encountered issue: {e}")

        # 6. Save normalized LCN images for presentation
        cv2.imwrite(os.path.join(out_dir, "lcn_ref.jpg"), img0_scaled)
        cv2.imwrite(os.path.join(out_dir, "lcn_target.jpg"), img1_scaled)

        # 7. Save Corresponding Match Points CSV with sensor-specific or standard header
        s0_label = sensor0.lower() if sensor0 else "ref"
        s1_label = sensor1.lower() if sensor1 else "target"
        csv_path = os.path.join(out_dir, "corresponding_match_points.csv")
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            f.write(f"x_ref_{s0_label},y_ref_{s0_label},x_target_{s1_label},y_target_{s1_label},reproj_residual_px\n")
            if res["success"] and len(res["orig_inliers0"]) > 0:
                residuals = res.get("residuals", np.zeros(len(res["orig_inliers0"])))
                for (x0, y0), (x1, y1), r in zip(res["orig_inliers0"], res["orig_inliers1"], residuals):
                    f.write(f"{x0:.4f},{y0:.4f},{x1:.4f},{y1:.4f},{r:.4f}\n")

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
            "sdi": res["sdi"],
            "checkpoint_rmse_px": res["checkpoint_rmse_px"],
            "checkpoint_rmse_meters": res["checkpoint_rmse_meters"],
            "ncc_cross_correlation": similarity_metrics["ncc"],
            "mutual_information": similarity_metrics["mutual_information"],
            "psnr_db": similarity_metrics["psnr"],
            "match_time_sec": res["elapsed_match_sec"],
            "total_time_sec": res["elapsed_total_sec"],
            "orig_inliers0": res["orig_inliers0"],
            "orig_inliers1": res["orig_inliers1"]
        }

        return output_metrics

