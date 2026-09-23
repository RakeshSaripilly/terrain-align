"""
task2pipeline.py / SunAngle/task2pipeline.py
TASK 2 PIPELINE - Full Task1+Task2 runner for SIH26166
Updated to use isro_metric_evaluator.py + explicit MAGSAC++

Flow:
1. Load lunar images (OHRC/TMC/IIRS vs LRO NAC)
2. MS-LCN + NGF preprocessing (your preprocessing.py)
3. Scale-space LoFTR matching (your matching_engine.py)
4. Sub-pixel LK refinement (your subpixel_uniformity.py)
5. Uniformity: Quadtree + ANMS (your subpixel_uniformity.py)
6. Robust MAGSAC++ homography (your geometry_warping.py - USAC_MAGSAC)
7. ISRO metrics evaluator (isro_metric_evaluator.py) -> 3 deliverables
8. Visuals: match_lines, checkerboard, heatmap, spatial_density

Usage:
python -m SunAngle.task2pipeline --low data/nac_low.png --high data/nac_high.png --out outputs_task2
"""

import os
import sys
import argparse
import json
from pathlib import Path
from typing import Optional, Tuple, Dict, Any
import cv2
import numpy as np

# Safe UTF-8 console output for Windows cp1252
if sys.stdout is not None and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# Ensure local LunarX directory is prioritized in sys.path
CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

# Imports with local and fallback resolution
try:
    from .pipeline import LunarCorrespondenceEngine
    from .isro_metric_evaluator import save_isro_deliverables, compute_isro_metrics
    from .evaluation import sift_baseline
    from .metrics_and_vis import visualize_matches, visualize_spatial_distribution
    from .geometry_warping import generate_checkerboard, compute_difference_map, warp_lunar_image
    from .preprocessing import infer_gsd_from_path
except (ImportError, ValueError):
    from pipeline import LunarCorrespondenceEngine
    from isro_metric_evaluator import save_isro_deliverables, compute_isro_metrics
    from evaluation import sift_baseline
    from metrics_and_vis import visualize_matches, visualize_spatial_distribution
    from geometry_warping import generate_checkerboard, compute_difference_map, warp_lunar_image
    from preprocessing import infer_gsd_from_path


def run_task2(
    low_sun_path: str,
    high_sun_path: str,
    out_dir: str = "SunAngle/outputs_task2",
    resize_long: int = 1024,
    conf_thresh: float = 0.20,
    ransac_thresh: float = 2.0,
    max_uniform_points: int = 1200,
    use_tps: bool = False,
    gsd_src: Optional[float] = None,
    gsd_ref: Optional[float] = None,
    k_min: int = 5,
    k_max: int = 25,
    model: str = "homography",
    engine: Optional[Any] = None
):
    os.makedirs(out_dir, exist_ok=True)

    if use_tps:
        model = "tps"

    # Infer GSD if not explicitly provided
    if gsd_ref is None:
        gsd_ref = infer_gsd_from_path(low_sun_path)
    if gsd_src is None:
        gsd_src = infer_gsd_from_path(high_sun_path)

    scale_ratio_str = f"{gsd_ref / gsd_src:.2f}x" if (gsd_ref and gsd_src) else "1.00x (unspecified)"

    print("\n" + "=" * 80)
    print(" SIH26166 TASK 2 PIPELINE - GSD SCALE-INVARIANT LoFTR + GRUEN LSM + MAGSAC++")
    print("=" * 80)
    print(f" Reference (low-sun)      : {low_sun_path} (GSD: {gsd_ref} m/px)")
    print(f" Target (high-sun)        : {high_sun_path} (GSD: {gsd_src} m/px)")
    print(f" Scale Ratio s (ref/src)  : {scale_ratio_str}")
    print(f" Geometric Model          : {model.upper()}")
    print(f" Output Dir               : {out_dir}")
    print(f" Resize Long              : {resize_long} (divisible by 8 for LoFTR)")
    print(f" LoFTR Conf Thresh        : {conf_thresh} (0.2 demo, 0.5 final)")
    print(f" MAGSAC++ Thresh          : {ransac_thresh}px")
    print("-" * 80)

    # 1. SIFT Baseline (Task1 - shows failure on lunar shadows)
    print("\n[STEP 1/6] Classical SIFT Baseline (expected to fail on lunar shadows)...")
    try:
        sift_matches = sift_baseline(low_sun_path, high_sun_path)
        print(f"   -> SIFT matches: {sift_matches} (typically <50 on sun-angle change)")
    except Exception as e:
        sift_matches = 0
        print(f"   -> SIFT error: {e}")

    # 2. Init Engine
    if engine is None:
        print("\n[STEP 2/6] Initializing LunarCorrespondenceEngine (MS-LCN + LoFTR outdoor)...")
        engine = LunarCorrespondenceEngine()
    else:
        print("\n[STEP 2/6] Reusing pre-initialized LunarCorrespondenceEngine...")
    print(f"   -> Device: {engine.matcher.device}")
    print(f"   -> Matcher: LoFTR outdoor (trained on MegaDepth - best for illumination)")

    # 3. Full Matching with GSD scale normalization, spatial quota, and subpixel Gruen LSM
    print("\n[STEP 3/6] Running GSD Scale Norm -> LoFTR -> Spatial Quota 8x8 -> Gruen LSM -> MAGSAC++...")
    print("   -> GSD Normalization: MTF Gaussian anti-aliasing filter + INTER_AREA downsampling")
    print("   -> Preprocessing: percentile stretch 1-99% + MS-LCN (25+71) + NGF")
    print(f"   -> Spatial Quotas: 8x8 grid (k_min={k_min}, k_max={k_max}) preventing crater rim bias")
    print("   -> Sub-pixel Refinement: Gruen Least Squares Matching (LSM) + cornerSubPix/LK fallback")
    print(f"   -> Robust Model: {model.upper()} with USAC_MAGSAC (conf 0.999)")

    result = engine.match(
        low_sun_path,
        high_sun_path,
        resize_long=resize_long,
        conf_thresh=conf_thresh,
        subpixel_refinement=True,
        enforce_uniformity=True,
        max_uniform_points=max_uniform_points,
        ransac_thresh=ransac_thresh,
        gsd0=gsd_ref,
        gsd1=gsd_src,
        k_min=k_min,
        k_max=k_max,
        model=model
    )

    print(f"\n   -> Raw LoFTR matches: {result['raw_matches']}")
    print(f"   -> After spatial quota binning: {result['uniform_matches']}")
    print(f"   -> Inliers: {result['inlier_count']} ({result['inlier_ratio']*100:.1f}%)")
    print(f"   -> Tie-point RMSE: {result['rmse_pixels']:.4f} px | Sub-pixel: {result['subpixel_accurate']}")
    print(f"   -> Checkpoint RMSE: {result['checkpoint_rmse_px']:.4f} px ({result['checkpoint_rmse_meters']:.4f} m)")
    print(f"   -> Spatial Distribution Index (SDI): {result['sdi']:.4f} (target >= 0.65)")
    print(f"   -> Coverage 3x3: {result['spatial_coverage']*100:.1f}% | Entropy: {result['spatial_entropy']:.3f}")

    # 4. ISRO Metrics Evaluator (Task2)
    print("\n[STEP 4/6] Running ISRO Metric Evaluator...")
    isro_metrics = save_isro_deliverables(
        result, out_dir=out_dir, save_registered=True,
        gsd_ref=result.get("gsd_ref"), gsd_src=result.get("gsd_src")
    )
    isro_metrics["sift_baseline_matches"] = sift_matches

    # 5. Visual Artifacts
    print("\n[STEP 5/6] Generating visual artifacts for deliverables...")
    img0_scaled = result["img0_scaled"]
    img1_scaled = result["img1_scaled"]
    pts0 = result["pts0_scaled"]
    pts1 = result["pts1_scaled"]
    inlier_mask = result["inlier_mask"]
    H = result["H_matrix"]

    # Match lines
    match_vis_path = os.path.join(out_dir, "match_lines.jpg")
    visualize_matches(img0_scaled, img1_scaled, pts0, pts1, inlier_mask, out_path=match_vis_path)
    print(f"   -> {match_vis_path}")

    # Spatial density
    density_path = os.path.join(out_dir, "spatial_density.jpg")
    inlier_pts0 = pts0[inlier_mask] if result["success"] else pts0
    visualize_spatial_distribution(img0_scaled.shape, inlier_pts0, out_path=density_path, grid_size=8)
    print(f"   -> {density_path}")

    if result["success"] and H is not None:
        try:
            H_inv = np.linalg.inv(H)
            img1_warped = warp_lunar_image(img1_scaled, img0_scaled.shape, H_inv)

            checker = generate_checkerboard(img0_scaled, img1_warped, square_size=64)
            checker_path = os.path.join(out_dir, "checkerboard_overlay.jpg")
            cv2.imwrite(checker_path, checker)
            print(f"   -> {checker_path}")

            _, heatmap = compute_difference_map(img0_scaled, img1_warped)
            heatmap_path = os.path.join(out_dir, "difference_heatmap.jpg")
            cv2.imwrite(heatmap_path, heatmap)
            print(f"   -> {heatmap_path}")

            # Save LCN images
            cv2.imwrite(os.path.join(out_dir, "lcn_ref.jpg"), img0_scaled)
            cv2.imwrite(os.path.join(out_dir, "lcn_target.jpg"), img1_scaled)

        except Exception as e:
            print(f"   -> Visual warp failed: {e}")

    # 6. Final JSON + Report
    print("\n[STEP 6/6] Saving final combined metrics...")
    final_metrics_path = os.path.join(out_dir, "metrics_task2_final.json")
    with open(final_metrics_path, "w", encoding="utf-8") as f:
        json.dump(isro_metrics, f, indent=2, default=lambda x: float(x) if isinstance(x, (np.float32, np.float64)) else x.tolist() if isinstance(x, np.ndarray) else x)
    print(f"   -> {final_metrics_path}")

    # Terminal Report
    isro_pass_text = "[PASS] YES" if isro_metrics['isro_pass'] else "[FAIL] NO"
    print("\n" + "=" * 80)
    print(" FINAL EVALUATION REPORT - SIH26166 TASK 2")
    print("=" * 80)
    print(f" {'Metric':<35} | {'Value':<20} | {'Target':<15}")
    print("-" * 80)
    print(f" {'SIFT Baseline':<35} | {sift_matches:<20} | {'<50 (Fails)':<15}")
    print(f" {'LoFTR Raw Matches':<35} | {isro_metrics['total_matches']:<20} | {'>1000':<15}")
    print(f" {'Uniform Matches (Quota)':<35} | {isro_metrics.get('uniform_matches', isro_metrics['total_matches']):<20} | {'500-1500':<15}")
    print(f" {'Inlier Count':<35} | {isro_metrics['inlier_count']:<20} | {'>500':<15}")
    print(f" {'Inlier Ratio':<35} | {isro_metrics['inlier_ratio']*100:.2f}%{'':<14} | {'>65%':<15}")
    print(f" {'Inliers <2px (Success)':<35} | {isro_metrics['inlier_count_2px']:<20} | {'>80%':<15}")
    print(f" {'Tie-point RMSE':<35} | {isro_metrics['rmse_pixels']:.4f} px{'':<12} | {'<0.5 px':<15}")
    print(f" {'Checkpoint RMSE':<35} | {isro_metrics['checkpoint_rmse_px']:.4f} px{'':<12} | {'<0.5 px':<15}")
    print(f" {'Checkpoint RMSE (Ground)':<35} | {isro_metrics['checkpoint_rmse_meters']:.4f} m{'':<13} | {'Sub-meter':<15}")
    print(f" {'Spatial Distribution Index (SDI)':<35} | {isro_metrics['sdi']:.4f}{'':<14} | {'>= 0.65':<15}")
    status_sub = "PASS" if isro_metrics['subpixel_accuracy'] else "FAIL"
    print(f" {'Sub-pixel Accuracy':<35} | {status_sub:<20} | {'REQUIRED':<15}")
    print(f" {'Coverage 3x3':<35} | {isro_metrics['uniformity_coverage_3x3']*100:.1f}%{'':<13} | {'>77%':<15}")
    print(f" {'Entropy 3x3':<35} | {isro_metrics['uniformity_entropy_3x3']:.3f}{'':<14} | {'>0.70':<15}")
    print(f" {'Scale Consistency':<35} | {'PASS' if isro_metrics['scale_consistency_pass'] else 'FAIL':<20} | {'s^2 +- 15%':<15}")
    print(f" {'ISRO PASS':<35} | {isro_pass_text:<20} | {'RMSE<2 & Cov>77%':<15}")
    print("=" * 80)
    print(f"\n[DELIVERABLES] in {out_dir}:")
    print(" 1. registered_product.jpg - warped target (Deliverable 1)")
    print(" 2. match_points.csv + mkpts*_inliers.npy + H.npy - match points (Deliverable 2)")
    print(" 3. metrics.json + isro_metrics.json - RMSE, inlier ratio, SDI, checkpoint error (Deliverable 3)")
    print(" 4. match_lines.jpg, checkerboard_overlay.jpg, difference_heatmap.jpg, spatial_density.jpg - PPT")
    print("=" * 80 + "\n")

    return isro_metrics


def main():
    parser = argparse.ArgumentParser(description="SIH26166 Task2 Pipeline - GSD Scale-Invariant Matching")
    parser.add_argument("--low", type=str, required=True, help="Reference low-sun image path (OHRC/TMC/LRO NAC)")
    parser.add_argument("--high", type=str, required=True, help="Target high-sun image path")
    parser.add_argument("--out", type=str, default="SunAngle/outputs_task2", help="Output directory")
    parser.add_argument("--resize", type=int, default=1024, help="Max dimension, divisible by 8")
    parser.add_argument("--conf", type=float, default=0.20, help="LoFTR confidence thresh")
    parser.add_argument("--ransac", type=float, default=2.0, help="MAGSAC++ reproj thresh px")
    parser.add_argument("--maxpts", type=int, default=1200, help="Max uniform points")
    parser.add_argument("--tps", action="store_true", help="Use TPS for non-rigid crater topography")
    parser.add_argument("--gsd_src", type=float, default=None, help="GSD of source/high-sun image in m/pixel")
    parser.add_argument("--gsd_ref", type=float, default=None, help="GSD of reference/low-sun image in m/pixel")
    parser.add_argument("--model", type=str, default="homography", choices=["homography", "affine", "tps"], help="Transformation model")
    parser.add_argument("--k_min", type=int, default=5, help="Minimum tie-points quota per grid cell")
    parser.add_argument("--k_max", type=int, default=25, help="Maximum tie-points quota per grid cell")
    args = parser.parse_args()

    run_task2(
        low_sun_path=args.low,
        high_sun_path=args.high,
        out_dir=args.out,
        resize_long=args.resize,
        conf_thresh=args.conf,
        ransac_thresh=args.ransac,
        max_uniform_points=args.maxpts,
        use_tps=args.tps,
        gsd_src=args.gsd_src,
        gsd_ref=args.gsd_ref,
        k_min=args.k_min,
        k_max=args.k_max,
        model=args.model
    )


if __name__ == "__main__":
    main()

