"""
isro_metric_evaluator.py / SunAngle/isro_metric_evaluator.py
TASK 2 - ISRO Deliverables Evaluator for SIH26166
Replaces old evaluation.py (SIFT baseline) with full PS compliance metrics

PS Requirements:
- RMSE, Inlier Match Count, Inlier Ratio @1px/@2px/@3px
- Success Rate <2px
- Uniform Distribution: 3x3 coverage >77% (7/9 cells) + entropy >0.7
- Sub-pixel accuracy: RMSE <0.5px target, <2.0px PASS
- Deliverables: registered_product.jpg, match_points.csv, metrics.json

Uses your geometry_warping.py USAC_MAGSAC (MAGSAC++) + subpixel_uniformity.py
"""

import os
import sys
import json
import csv
from typing import Dict, Any, Tuple, Optional
import cv2
import numpy as np

# Safe UTF-8 console handling for Windows cp1252
if sys.stdout is not None and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# Try import from package, fallback to flat
try:
    from .subpixel_uniformity import compute_spatial_uniformity_metrics, compute_sdi_metric
    from .geometry_warping import warp_lunar_image, estimate_robust_transformation, compute_scale_consistency
    from .evaluation import compute_checkpoint_rmse
except (ImportError, ValueError):
    from subpixel_uniformity import compute_spatial_uniformity_metrics, compute_sdi_metric
    from geometry_warping import warp_lunar_image, estimate_robust_transformation, compute_scale_consistency
    from evaluation import compute_checkpoint_rmse


def compute_reprojection_errors(pts0: np.ndarray, pts1: np.ndarray, H: np.ndarray) -> np.ndarray:
    """Forward reprojection error || H*pts0 - pts1 ||"""
    if H is None or len(pts0) == 0:
        return np.array([999.0] * len(pts0)) if len(pts0) > 0 else np.array([])
    pts0_t = cv2.perspectiveTransform(pts0.reshape(-1, 1, 2).astype(np.float32), H.astype(np.float64)).reshape(-1, 2)
    return np.linalg.norm(pts0_t - pts1, axis=1)


def compute_isro_metrics(
    pts0_all: np.ndarray,
    pts1_all: np.ndarray,
    pts0_inliers: np.ndarray,
    pts1_inliers: np.ndarray,
    H: np.ndarray,
    img_shape: Tuple[int, int],
    residuals_inliers: Optional[np.ndarray] = None,
    conf_all: Optional[np.ndarray] = None,
    gsd_ref: Optional[float] = None,
    gsd_src: Optional[float] = None
) -> Dict[str, Any]:
    """
    Computes complete ISRO PS metrics for Task 2 including:
    - Spatial Distribution Index (SDI) on 8x8 grid
    - 80/20 Tie-Point / Check-Point validation split RMSE (px and meters)
    - Scale consistency check (Jacobian det |J| vs s^2 +- 15%)
    - Sub-pixel accuracy and uniform distribution metrics
    """
    total = int(len(pts0_all))
    inlier_n = int(len(pts0_inliers))

    # Errors for ALL points using H
    err_all = compute_reprojection_errors(pts0_all, pts1_all, H) if H is not None else np.array([])

    def count_at(th: float) -> int:
        return int(np.sum(err_all < th)) if len(err_all) > 0 else 0

    def ratio_at(th: float) -> float:
        return float(np.mean(err_all < th)) if len(err_all) > 0 else 0.0

    # RMSE from inliers (true sub-pixel metric)
    if residuals_inliers is not None and len(residuals_inliers) > 0:
        rmse = float(np.sqrt(np.mean(residuals_inliers ** 2)))
        mean_err = float(np.mean(residuals_inliers))
        median_err = float(np.median(residuals_inliers))
        max_err = float(np.max(residuals_inliers))
    elif inlier_n > 0 and H is not None:
        err_in = compute_reprojection_errors(pts0_inliers, pts1_inliers, H)
        rmse = float(np.sqrt(np.mean(err_in ** 2)))
        mean_err = float(np.mean(err_in))
        median_err = float(np.median(err_in))
        max_err = float(np.max(err_in))
    else:
        rmse = 999.0
        mean_err = 999.0
        median_err = 999.0
        max_err = 999.0

    # Uniformity & Spatial Distribution Index (SDI)
    pts_for_uniformity = pts0_inliers if inlier_n > 0 else pts0_all

    uniformity_3x3 = compute_spatial_uniformity_metrics(pts_for_uniformity, img_shape, grid_divisions=3)
    uniformity_6x6 = compute_spatial_uniformity_metrics(pts_for_uniformity, img_shape, grid_divisions=6)
    uniformity_8x8 = compute_spatial_uniformity_metrics(pts_for_uniformity, img_shape, grid_divisions=8)
    sdi_8x8 = compute_sdi_metric(pts_for_uniformity, img_shape, grid_divisions=8)

    # 80/20 Tie-Point vs Check-Point Reprojection RMSE
    chk_metrics = compute_checkpoint_rmse(pts0_inliers, pts1_inliers, H, split_ratio=0.8, gsd_ref=gsd_ref)

    # Scale consistency check
    scale_ratio = float(gsd_ref / gsd_src) if (gsd_ref is not None and gsd_src is not None and gsd_src > 0) else 1.0
    scale_check = compute_scale_consistency(H, expected_scale_ratio=scale_ratio)

    # Inlier ratio
    inlier_ratio = inlier_n / total if total > 0 else 0.0

    metrics = {
        # Basic counts
        "total_matches": total,
        "inlier_count": inlier_n,
        "inlier_ratio": round(float(inlier_ratio), 4),

        # Threshold metrics (PS: inlier match count, inlier ratio)
        "inlier_count_1px": count_at(1.0),
        "inlier_ratio_1px": round(ratio_at(1.0), 4),
        "inlier_count_2px": count_at(2.0),
        "inlier_ratio_2px": round(ratio_at(2.0), 4),
        "success_rate_2px": round(ratio_at(2.0), 4),
        "inlier_count_3px": count_at(3.0),
        "inlier_ratio_3px": round(ratio_at(3.0), 4),

        # RMSE metrics
        "rmse_pixels": round(float(rmse), 4),
        "mean_error_pixels": round(float(mean_err), 4),
        "median_error_pixels": round(float(median_err), 4),
        "max_error_pixels": round(float(max_err), 4),
        "subpixel_accuracy": bool(rmse < 1.0),
        "subpixel_05": bool(rmse < 0.5),
        "rmse_target_05": "<0.5 px target",
        "rmse_pass_2px": bool(rmse < 2.0),

        # 80/20 Checkpoint RMSE (ISRO SAC Requirement)
        "checkpoint_count": chk_metrics["checkpoint_count"],
        "tiepoint_count": chk_metrics["tiepoint_count"],
        "checkpoint_rmse_px": chk_metrics["checkpoint_rmse_px"],
        "checkpoint_rmse_meters": chk_metrics["checkpoint_rmse_meters"],
        "tiepoint_rmse_px": chk_metrics["tiepoint_rmse_px"],
        "tiepoint_rmse_meters": chk_metrics["tiepoint_rmse_meters"],

        # Spatial Distribution Index (SDI)
        "sdi": sdi_8x8,
        "sdi_8x8": sdi_8x8,
        "sdi_pass": bool(sdi_8x8 >= 0.65),

        # Uniformity metrics
        "uniformity_coverage_3x3": uniformity_3x3["coverage_fraction"],
        "uniformity_entropy_3x3": uniformity_3x3["spatial_entropy"],
        "uniformity_hist_3x3": uniformity_3x3.get("cell_counts", uniformity_3x3.get("hist_2d")),
        "uniformity_occupied_3x3": uniformity_3x3["occupied_cells"],
        "uniformity_empty_3x3": uniformity_3x3["total_cells"] - uniformity_3x3["occupied_cells"],

        "uniformity_coverage_6x6": uniformity_6x6["coverage_fraction"],
        "uniformity_entropy_6x6": uniformity_6x6["spatial_entropy"],

        "uniformity_coverage_8x8": uniformity_8x8["coverage_fraction"],
        "uniformity_entropy_8x8": uniformity_8x8["spatial_entropy"],

        # Scale Consistency
        "scale_ratio": round(scale_ratio, 4),
        "gsd_src": gsd_src,
        "gsd_ref": gsd_ref,
        "scale_consistency_pass": scale_check["scale_consistency_pass"],
        "scale_consistency_det": scale_check["det_J"],
        "scale_expected_sq": scale_check["expected_scale_sq"],
        "scale_error_ratio": scale_check["scale_error_ratio"],

        # ISRO final PASS/FAIL
        "isro_pass_rmse": bool(rmse < 2.0),
        "isro_pass_coverage": bool(uniformity_3x3["coverage_fraction"] >= 0.77),
        "isro_pass": bool(rmse < 2.0 and uniformity_3x3["coverage_fraction"] >= 0.77),

        # Metadata
        "method_robust": "USAC_MAGSAC (MAGSAC++) with fallback RANSAC",
        "method_refinement": "Gruen Least Squares Matching (LSM) + cornerSubPix / LK fallback",
        "method_uniformity": "Spatial Grid Quota Binning 8x8 (k_min>=5, k_max<=25) + ANMS"
    }

    return metrics


def save_match_points_csv(
    pts0: np.ndarray,
    pts1: np.ndarray,
    conf: np.ndarray,
    inlier_mask: np.ndarray,
    out_path: str
):
    """ISRO Deliverable 2: match points x_src,y_src,x_ref,y_ref,confidence,is_inlier"""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["x_src", "y_src", "x_ref", "y_ref", "confidence", "is_inlier"])
        for (x0, y0), (x1, y1), c, inl in zip(pts0, pts1, conf, inlier_mask):
            writer.writerow([f"{x0:.4f}", f"{y0:.4f}", f"{x1:.4f}", f"{y1:.4f}", f"{c:.4f}", int(inl)])
    print(f"[ISRO-METRIC] Saved {len(pts0)} points -> {out_path}")


def save_isro_deliverables(
    pipeline_result: Dict[str, Any],
    out_dir: str = "outputs_task2",
    save_registered: bool = True,
    gsd_ref: Optional[float] = None,
    gsd_src: Optional[float] = None
) -> Dict[str, Any]:
    """
    Takes result from LunarCorrespondenceEngine.match() or register_and_visualize()
    Generates all 3 ISRO deliverables
    """
    os.makedirs(out_dir, exist_ok=True)

    pts0 = pipeline_result["pts0_scaled"]
    pts1 = pipeline_result["pts1_scaled"]
    inlier_mask = pipeline_result["inlier_mask"]
    H = np.array(pipeline_result["H_scaled"]) if pipeline_result.get("H_scaled") is not None else None
    img0_scaled = pipeline_result["img0_scaled"]
    img1_scaled = pipeline_result["img1_scaled"]

    pts0_in = pts0[inlier_mask] if pipeline_result.get("success", False) else np.array([]).reshape(0, 2)
    pts1_in = pts1[inlier_mask] if pipeline_result.get("success", False) else np.array([]).reshape(0, 2)

    # Residuals for RMSE
    if H is not None and len(pts0_in) > 0:
        pts0_in_t = cv2.perspectiveTransform(pts0_in.reshape(-1, 1, 2).astype(np.float32), H.astype(np.float64)).reshape(-1, 2)
        residuals = np.linalg.norm(pts0_in_t - pts1_in, axis=1)
    else:
        residuals = np.array([])

    conf = np.ones(len(pts0), dtype=np.float32)
    gsd_r = gsd_ref if gsd_ref is not None else pipeline_result.get("gsd_ref")
    gsd_s = gsd_src if gsd_src is not None else pipeline_result.get("gsd_src")

    # Compute full metrics
    metrics = compute_isro_metrics(
        pts0, pts1, pts0_in, pts1_in, H, img0_scaled.shape,
        residuals_inliers=residuals, conf_all=conf,
        gsd_ref=gsd_r, gsd_src=gsd_s
    )

    # Add pipeline timings and counts
    metrics.update({
        "raw_matches": pipeline_result.get("raw_matches", len(pts0)),
        "uniform_matches": pipeline_result.get("uniform_matches", len(pts0)),
        "scale0": pipeline_result.get("scale0", 1.0),
        "scale1": pipeline_result.get("scale1", 1.0),
        "elapsed_total_sec": pipeline_result.get("elapsed_total_sec", 0.0),
        "elapsed_match_sec": pipeline_result.get("elapsed_match_sec", 0.0),
    })

    # Save H
    if H is not None:
        np.save(os.path.join(out_dir, "H_scaled.npy"), H)
        np.save(os.path.join(out_dir, "H.npy"), H)
        with open(os.path.join(out_dir, "H_matrix.txt"), "w", encoding='utf-8') as f:
            f.write(f"# Homography matrix (scaled coords) - maps img0_scaled -> img1_scaled\n")
            for row in H:
                f.write(" ".join([f"{v:.8f}" for v in row]) + "\n")

    # Save match points (Deliverable 2)
    save_match_points_csv(pts0, pts1, conf, inlier_mask, os.path.join(out_dir, "match_points.csv"))
    np.save(os.path.join(out_dir, "mkpts0_all.npy"), pts0)
    np.save(os.path.join(out_dir, "mkpts1_all.npy"), pts1)
    np.save(os.path.join(out_dir, "mkpts0_inliers.npy"), pts0_in)
    np.save(os.path.join(out_dir, "mkpts1_inliers.npy"), pts1_in)
    np.save(os.path.join(out_dir, "inlier_mask.npy"), inlier_mask)

    # Save registered product (Deliverable 1)
    if save_registered and H is not None:
        try:
            H_inv = np.linalg.inv(H)
            registered = warp_lunar_image(img1_scaled, img0_scaled.shape, H_inv)
            cv2.imwrite(os.path.join(out_dir, "registered_product.jpg"), registered)
            cv2.imwrite(os.path.join(out_dir, "registered_target.jpg"), registered)
            print(f"[ISRO-METRIC] Saved registered_product.jpg -> {out_dir}")
        except Exception as e:
            print(f"[ISRO-METRIC] Warp failed: {e}")

    # Save metrics.json (Deliverable 3)
    with open(os.path.join(out_dir, "metrics.json"), "w", encoding='utf-8') as f:
        json.dump(metrics, f, indent=2, default=lambda x: x.tolist() if isinstance(x, np.ndarray) else float(x) if isinstance(x, (np.float32, np.float64)) else x)

    with open(os.path.join(out_dir, "isro_metrics.json"), "w", encoding='utf-8') as f:
        json.dump(metrics, f, indent=2, default=lambda x: x.tolist() if isinstance(x, np.ndarray) else float(x) if isinstance(x, (np.float32, np.float64)) else x)

    # Print dashboard
    isro_pass_text = "[PASS] YES" if metrics['isro_pass'] else "[FAIL] NO"
    print("\n" + "=" * 78)
    print(" ISRO METRIC EVALUATOR - TASK 2 DASHBOARD")
    print("=" * 78)
    print(f" Total Matches          : {metrics['total_matches']}")
    print(f" Inlier Count           : {metrics['inlier_count']} ({metrics['inlier_ratio']*100:.1f}%)")
    print(f" Inliers <1px           : {metrics['inlier_count_1px']} ({metrics['inlier_ratio_1px']*100:.1f}%)")
    print(f" Inliers <2px [PS]      : {metrics['inlier_count_2px']} ({metrics['inlier_ratio_2px']*100:.1f}%) <- Success Rate")
    print(f" Inliers <3px           : {metrics['inlier_count_3px']} ({metrics['inlier_ratio_3px']*100:.1f}%)")
    print(f" RMSE                   : {metrics['rmse_pixels']:.4f} px (target <0.5, pass <2.0)")
    print(f" Mean Error             : {metrics['mean_error_pixels']:.4f} px")
    print(f" Checkpoint RMSE        : {metrics['checkpoint_rmse_px']:.4f} px ({metrics['checkpoint_rmse_meters']:.4f} m)")
    print(f" Spatial Distribution (SDI): {metrics['sdi']:.4f} (target >= 0.65)")
    print(f" Coverage 3x3           : {metrics['uniformity_coverage_3x3']*100:.1f}% (target >77% = 7/9 cells)")
    print(f" Entropy 3x3            : {metrics['uniformity_entropy_3x3']:.3f} (target >0.7)")
    print(f" Coverage 8x8           : {metrics['uniformity_coverage_8x8']*100:.1f}%")
    print(f" Scale Consistency      : {'PASS' if metrics['scale_consistency_pass'] else 'FAIL'} (|J|={metrics['scale_consistency_det']:.3f}, exp={metrics['scale_expected_sq']:.3f})")
    print(f" Robust Method          : {metrics['method_robust']}")
    print(f" ISRO PASS              : {isro_pass_text}")
    print("=" * 78 + "\n")

    return metrics



# Standalone test if run directly
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="ISRO Metric Evaluator - Task2")
    parser.add_argument("--pts0", type=str, help="npy file mkpts0")
    parser.add_argument("--pts1", type=str, help="npy file mkpts1")
    parser.add_argument("--H", type=str, help="npy file H")
    parser.add_argument("--img", type=str, help="reference image for shape")
    parser.add_argument("--out", type=str, default="outputs_task2")
    args = parser.parse_args()

    if args.pts0 and args.pts1 and args.H:
        pts0 = np.load(args.pts0)
        pts1 = np.load(args.pts1)
        H = np.load(args.H)
        # dummy result dict
        result = {
            "pts0_scaled": pts0,
            "pts1_scaled": pts1,
            "inlier_mask": np.ones(len(pts0), dtype=bool),
            "H_scaled": H,
            "img0_scaled": np.zeros((1024, 1024), dtype=np.uint8),
            "img1_scaled": np.zeros((1024, 1024), dtype=np.uint8),
            "raw_matches": len(pts0),
            "uniform_matches": len(pts0),
            "scale0": 1.0,
            "scale1": 1.0,
            "success": True
        }
        save_isro_deliverables(result, out_dir=args.out)
    else:
        print("Provide --pts0, --pts1, --H or use via pipeline")
