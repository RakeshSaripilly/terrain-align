"""
task2pipeline.py / SunAngle/task2pipeline.py
TASK 2 PIPELINE - Full Task1+Task2 runner
Updated to use isro_metric_evaluator.py + explicit MAGSAC++
"""

import os
import sys
import argparse
import json
from pathlib import Path
import cv2
import numpy as np

# Ensure project root in path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Imports from your Task1 code
try:
    from SunAngle.pipeline import LunarCorrespondenceEngine
    from SunAngle.isro_metric_evaluator import save_isro_deliverables, compute_isro_metrics
    from SunAngle.evaluation import sift_baseline
    from SunAngle.metrics_and_vis import visualize_matches, visualize_spatial_distribution
    from SunAngle.geometry_warping import generate_checkerboard, compute_difference_map, warp_lunar_image
except ImportError:
    # Fallback flat import
    from pipeline import LunarCorrespondenceEngine
    from isro_metric_evaluator import save_isro_deliverables
    from evaluation import sift_baseline
    from metrics_and_vis import visualize_matches, visualize_spatial_distribution
    from geometry_warping import generate_checkerboard, compute_difference_map, warp_lunar_image


def run_task2(
    low_sun_path: str,
    high_sun_path: str,
    out_dir: str = "SunAngle/outputs_task2",
    resize_long: int = 1024,
    conf_thresh: float = 0.20,
    ransac_thresh: float = 2.0,
    max_uniform_points: int = 1200,
    use_tps: bool = False
):
    os.makedirs(out_dir, exist_ok=True)

    print("\n" + "=" * 80)
    print(" TASK 2 PIPELINE - LoFTR + MS-LCN + LK + MAGSAC++ + EVALUATION METRICS")
    print("=" * 80)
    print(f" Reference (low-sun)      : {low_sun_path}")
    print(f" Target (high-sun)        : {high_sun_path}")
    print(f" Output Dir               : {out_dir}")
    print(f" Resize Long              : {resize_long} (divisible by 8 for LoFTR)")
    print(f" LoFTR Conf Thresh        : {conf_thresh} (0.2 demo, 0.5 final)")
    print(f" MAGSAC++ Thresh          : {ransac_thresh}px")
    print("-" * 80)

    # 1. SIFT Baseline (Task1 - shows failure)
    print("\n[STEP 1/6] Classical SIFT Baseline (expected to fail on lunar shadows)...")
    try:
        sift_matches = sift_baseline(low_sun_path, high_sun_path)
        print(f"   -> SIFT matches: {sift_matches} (typically <50 on sun-angle change)")
    except Exception as e:
        sift_matches = 0
        print(f"   -> SIFT error: {e}")

    # 2. Init Engine (Task1 - LoFTR + MS-LCN)
    print("\n[STEP 2/6] Initializing LunarCorrespondenceEngine (MS-LCN + LoFTR outdoor)...")
    engine = LunarCorrespondenceEngine()
    print(f"   -> Device: {engine.matcher.device}")
    print(f"   -> Matcher: LoFTR outdoor (trained on MegaDepth - best for illumination)")

    # 3. Full Matching (Task1+2)
    print("\n[STEP 3/6] Running MS-LCN -> LoFTR -> Quadtree binning -> Sub-pixel LK -> MAGSAC++...")
    print("   -> Preprocessing: percentile stretch 1-99% + MS-LCN (25+71) + NGF")
    print("   -> Matching: Scale-space pyramid [1.0,0.5,0.25] for OHRC/TMC/IIRS")
    print("   -> Uniformity: Quadtree 8x8 + ANMS")
    print("   -> Refinement: Inverse-Compositional LK + cornerSubPix")
    print("   -> Robust: USAC_MAGSAC (MAGSAC++) 15000 iters, conf 0.999")

    result = engine.match(
        low_sun_path,
        high_sun_path,
        resize_long=resize_long,
        conf_thresh=conf_thresh,
        subpixel_refinement=True,
        enforce_uniformity=True,
        max_uniform_points=max_uniform_points,
        ransac_thresh=ransac_thresh
    )

    print(f"\n   -> Raw LoFTR matches: {result['raw_matches']}")
    print(f"   -> After uniformity: {result['uniform_matches']}")
    print(f"   -> MAGSAC++ inliers: {result['inlier_count']} ({result['inlier_ratio']*100:.1f}%)")
    print(f"   -> RMSE: {result['rmse_pixels']:.4f}px | Sub-pixel: {result['subpixel_accurate']}")
    print(f"   -> Coverage: {result['spatial_coverage']*100:.1f}% | Entropy: {result['spatial_entropy']:.3f}")

    # 4. ISRO Metrics Evaluator (Task2)
    print("\n[STEP 4/6] Running ISRO Metric Evaluator (Task2 evaluator)...")
    isro_metrics = save_isro_deliverables(result, out_dir=out_dir, save_registered=True)
    isro_metrics["sift_baseline_matches"] = sift_matches

    # 5. Visual Artifacts (for PPT - Task2)
    print("\n[STEP 5/6] Generating visual artifacts for PPT...")
    img0_scaled = result["img0_scaled"]
    img1_scaled = result["img1_scaled"]
    pts0 = result["pts0_scaled"]
    pts1 = result["pts1_scaled"]
    inlier_mask = result["inlier_mask"]
    H = np.array(result["H_scaled"]) if result["H_scaled"] is not None else None

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
        # Warp and checkerboard
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
    with open(final_metrics_path, "w") as f:
        json.dump(isro_metrics, f, indent=2, default=lambda x: float(x) if isinstance(x, (np.float32, np.float64)) else x.tolist() if isinstance(x, np.ndarray) else x)
    print(f"   -> {final_metrics_path}")

    # Terminal Report
    print("\n" + "=" * 80)
    print(" FINAL EVALUATION REPORT - TASK 2")
    print("=" * 80)
    print(f" {'Metric':<35} | {'Value':<20} | {'Target':<15}")
    print("-" * 80)
    print(f" {'SIFT Baseline':<35} | {sift_matches:<20} | {'<50 (Fails)':<15}")
    print(f" {'LoFTR Raw Matches':<35} | {isro_metrics['total_matches']:<20} | {'>1000':<15}")
    print(f" {'Uniform Matches':<35} | {isro_metrics.get('uniform_matches', isro_metrics['total_matches']):<20} | {'500-1500':<15}")
    print(f" {'MAGSAC++ Inliers':<35} | {isro_metrics['inlier_count']:<20} | {'>500':<15}")
    print(f" {'Inlier Ratio':<35} | {isro_metrics['inlier_ratio']*100:.2f}%{'':<14} | {'>80%':<15}")
    print(f" {'Inliers <2px (Success)':<35} | {isro_metrics['inlier_count_2px']:<20} | {'>80%':<15}")
    print(f" {'RMSE':<35} | {isro_metrics['rmse_pixels']:.4f} px{'':<12} | {'<0.5 px':<15}")
    status_sub = "PASS" if isro_metrics['subpixel_accuracy'] else "FAIL"
    print(f" {'Sub-pixel <1px':<35} | {status_sub:<20} | {'REQUIRED':<15}")
    print(f" {'Coverage 3x3':<35} | {isro_metrics['uniformity_coverage_3x3']*100:.1f}%{'':<13} | {'>77%':<15}")
    print(f" {'Entropy 3x3':<35} | {isro_metrics['uniformity_entropy_3x3']:.3f}{'':<14} | {'>0.70':<15}")
    print(f" {'BENCHMARK PASS':<35} | {'✅ YES' if isro_metrics['isro_pass'] else '❌ NO':<20} | {'RMSE<2 & Cov>77%':<15}")
    print("=" * 80)
    print(f"\n[DELIVERABLES] in {out_dir}:")
    print(" 1. registered_product.jpg - warped source (Deliverable 1)")
    print(" 2. match_points.csv + mkpts*_inliers.npy + H.npy - match points (Deliverable 2)")
    print(" 3. metrics.json + isro_metrics.json - RMSE, inlier ratio, uniformity (Deliverable 3)")
    print(" 4. match_lines.jpg, checkerboard_overlay.jpg, difference_heatmap.jpg, spatial_density.jpg - Visuals")
    print("=" * 80 + "\n")

    return isro_metrics


def main():
    parser = argparse.ArgumentParser(description="Task2 Pipeline - Full Evaluation Suite")
    parser.add_argument("--low", type=str, required=True, help="Reference low-sun image path (OHRC/TMC/LRO NAC)")
    parser.add_argument("--high", type=str, required=True, help="Target high-sun image path")
    parser.add_argument("--out", type=str, default="SunAngle/outputs_task2", help="Output directory")
    parser.add_argument("--resize", type=int, default=1024, help="Max dimension, divisible by 8")
    parser.add_argument("--conf", type=float, default=0.20, help="LoFTR confidence thresh")
    parser.add_argument("--ransac", type=float, default=2.0, help="MAGSAC++ reproj thresh px")
    parser.add_argument("--maxpts", type=int, default=1200, help="Max uniform points")
    parser.add_argument("--tps", action="store_true", help="Use TPS for non-rigid crater topography")
    args = parser.parse_args()

    run_task2(
        low_sun_path=args.low,
        high_sun_path=args.high,
        out_dir=args.out,
        resize_long=args.resize,
        conf_thresh=args.conf,
        ransac_thresh=args.ransac,
        max_uniform_points=args.maxpts,
        use_tps=args.tps
    )


if __name__ == "__main__":
    main()
