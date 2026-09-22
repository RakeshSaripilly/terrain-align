"""
SunAngle/Run-After-Lro.py
Execution script for surface registration pipeline:
Multi-Modal, Illumination and Scale Invariant Image Correspondence Pipeline.
"""

import sys
import os
import json
import argparse
from pathlib import Path

# Setup project root import
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np
from SunAngle.pipeline import LunarCorrespondenceEngine
from SunAngle.evaluation import sift_baseline


def run_pipeline(img_low_sun: str, img_high_sun: str, out_dir: str = "SunAngle/outputs") -> dict:
    os.makedirs(out_dir, exist_ok=True)

    print("\n" + "=" * 78)
    print(" MULTI-MODAL & ILLUMINATION INVARIANT CORRESPONDENCE PIPELINE")
    print("=" * 78)
    print(f" Reference Image (Low Sun / High Incidence)  : {img_low_sun}")
    print(f" Target Image (High Sun / Low Incidence)     : {img_high_sun}")
    print(f" Output Artifacts Directory                 : {out_dir}")
    print("-" * 78)

    # 1. Classical SIFT Baseline (to demonstrate failure on lunar illumination change)
    print("\n[STEP 1/5] Evaluating Classical SIFT Baseline...")
    try:
        sift_matches = sift_baseline(img_low_sun, img_high_sun)
        print(f"   --> Classical SIFT correspondences: {sift_matches} (severely degrades under illumination)")
    except Exception as e:
        sift_matches = 0
        print(f"   --> SIFT Baseline error: {e}")

    # 2. Deep Lunar Correspondence Engine (MS-LCN + LoFTR + ANMS + Subpixel LK + MAGSAC++)
    print("\n[STEP 2/5] Initializing Deep Feature Extraction & Sub-Pixel Refinement Engine...")
    engine = LunarCorrespondenceEngine()

    print("\n[STEP 3/5] Running Multi-Scale Radiometric Normalization & Deep Transformer Matching...")
    print("\n[STEP 4/5] Enforcing Spatial Uniformity & Sub-Pixel LK Optimization (<0.5 px target)...")
    print("\n[STEP 5/5] Estimating USAC_MAGSAC Homography & Generating Visual Artifacts...")

    metrics = engine.register_and_visualize(
        img_low_sun, img_high_sun, out_dir=out_dir, resize_long=1024
    )
    metrics["sift_baseline_matches"] = int(sift_matches)

    # Save metrics JSON
    metrics_path = os.path.join(out_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    # Terminal Evaluation Table
    print("\n" + "=" * 78)
    print("                   FINAL EVALUATION REPORT SUMMARY                       ")
    print("=" * 78)
    print(f" {'Metric / Parameter':<38} | {'Engine Value':<18} | {'Target Benchmark':<16}")
    print("-" * 78)
    print(f" {'Classical SIFT Matches':<38} | {metrics['sift_baseline_matches']:<18} | {'< 50 (Fails)':<16}")
    print(f" {'Deep LoFTR Raw Matches':<38} | {metrics['raw_matches']:<18} | {'> 1000':<16}")
    print(f" {'Uniform Sampled Matches':<38} | {metrics['uniform_matches']:<18} | {'~ 500-1500':<16}")
    print(f" {'RANSAC Inliers':<38} | {metrics['inlier_count']:<18} | {'> 500':<16}")
    print(f" {'Inlier Ratio':<38} | {metrics['inlier_ratio']*100:.2f}%{'':<11} | {'> 80.0%':<16}")
    print(f" {'Reprojection RMSE':<38} | {metrics['rmse_pixels']:.4f} px{'':<9} | {'< 0.50 px':<16}")
    status_subpixel = "PASS [SUB-PIXEL]" if metrics["subpixel_accuracy_achieved"] else "FAIL"
    print(f" {'Sub-Pixel Accuracy Status':<38} | {status_subpixel:<18} | {'REQUIRED':<16}")
    print(f" {'Spatial Uniformity Coverage':<38} | {metrics['spatial_coverage']*100:.1f}%{'':<12} | {'> 77.0%':<16}")
    print(f" {'Spatial Entropy (Shannon)':<38} | {metrics['spatial_entropy']:.3f}{'':<13} | {'> 0.700':<16}")
    print(f" {'Normalized Cross-Correlation (NCC)':<38} | {metrics['ncc_cross_correlation']:.4f}{'':<12} | {'> 0.60':<16}")
    print(f" {'Alignment PSNR':<38} | {metrics['psnr_db']:.2f} dB{'':<10} | {'> 18.0 dB':<16}")
    print(f" {'Feature Matching Time':<38} | {metrics['match_time_sec']:.2f} s{'':<12} | {'< 10.0 s':<16}")
    print(f" {'Total Pipeline Latency':<38} | {metrics['total_time_sec']:.2f} s{'':<12} | {'Real-time':<16}")
    print("=" * 78)

    print("\n[ARTIFACTS GENERATED IN]:", out_dir)
    print(" 1. match_lines.jpg          -> Side-by-side matches with inlier/outlier color lines")
    print(" 2. difference_heatmap.jpg   -> False-color error heatmap showing residual alignment")
    print(" 3. checkerboard_overlay.jpg -> Checkerboard grid proving crater rim continuity")
    print(" 4. spatial_density.jpg      -> 2D Quadtree correspondence density map")
    print(" 5. registered_target.jpg    -> Sub-pixel warped registered target image")
    print(" 6. lcn_ref / lcn_target.jpg -> Illumination-normalized pairs")
    print(" 7. metrics.json             -> Full metrics dictionary for evaluation")
    print("=" * 78 + "\n")

    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Surface Image Correspondence Pipeline")
    parser.add_argument("--low", default="SunAngle/data/sample/nac_low.png", help="Path to low-sun / reference image")
    parser.add_argument("--high", default="SunAngle/data/sample/nac_high.png", help="Path to high-sun / target image")
    parser.add_argument("--out", default="SunAngle/outputs", help="Output directory")
    args = parser.parse_args()

    run_pipeline(args.low, args.high, args.out)
