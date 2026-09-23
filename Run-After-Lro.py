"""
SunAngle/Run-After-Lro.py
Execution script for SIH26166:
Multi-Modal, Sun Angle and Scale Invariant Lunar Image Correspondence Pipeline.
Chandrayaan-2 (OHRC, TMC, IIRS) & LRO NAC datasets.
"""

import sys
import os
import json
import argparse
from pathlib import Path
from typing import Optional, Dict, Any, Tuple

# Safe UTF-8 console output for Windows cp1252
if sys.stdout is not None and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

import cv2
import numpy as np

try:
    from .pipeline import LunarCorrespondenceEngine
    from .evaluation import sift_baseline
    from .preprocessing import infer_gsd_from_path
except (ImportError, ValueError):
    from pipeline import LunarCorrespondenceEngine
    from evaluation import sift_baseline
    from preprocessing import infer_gsd_from_path


def run_pipeline(
    img_low_sun: str,
    img_high_sun: str,
    out_dir: str = "SunAngle/outputs",
    gsd_src: Optional[float] = None,
    gsd_ref: Optional[float] = None,
    model: str = "homography"
) -> dict:
    os.makedirs(out_dir, exist_ok=True)

    # Infer GSD if not provided
    if gsd_ref is None:
        gsd_ref = infer_gsd_from_path(img_low_sun)
    if gsd_src is None:
        gsd_src = infer_gsd_from_path(img_high_sun)

    scale_ratio = (gsd_ref / gsd_src) if (gsd_ref and gsd_src) else 1.0

    print("\n" + "=" * 78)
    print(" SIH26166: LUNAR MULTI-MODAL & SUN ANGLE INVARIANT CORRESPONDENCE PIPELINE")
    print("=" * 78)
    print(f" Reference Image (Low Sun / High Incidence)  : {img_low_sun} (GSD: {gsd_ref} m/px)")
    print(f" Target Image (High Sun / Low Incidence)     : {img_high_sun} (GSD: {gsd_src} m/px)")
    print(f" Scale Ratio s (ref/src)                     : {scale_ratio:.2f}x")
    print(f" Output Artifacts Directory                 : {out_dir}")
    print("-" * 78)

    # 1. Classical SIFT Baseline
    print("\n[STEP 1/5] Evaluating Classical SIFT Baseline...")
    try:
        sift_matches = sift_baseline(img_low_sun, img_high_sun)
        print(f"   --> Classical SIFT correspondences: {sift_matches} (severely degrades under illumination)")
    except Exception as e:
        sift_matches = 0
        print(f"   --> SIFT Baseline error: {e}")

    # 2. Deep Lunar Correspondence Engine
    print("\n[STEP 2/5] Initializing GSD-Aware Lunar Correspondence Engine...")
    engine = LunarCorrespondenceEngine()

    print("\n[STEP 3/5] Running MTF Anti-Aliasing & GSD Scale Normalization...")
    print("\n[STEP 4/5] Enforcing 8x8 Spatial Quota Binning & Gruen LSM / Native Refinement...")
    print("\n[STEP 5/5] Estimating Robust Transformation & Generating Visual Deliverables...")

    metrics = engine.register_and_visualize(
        img_low_sun, img_high_sun,
        out_dir=out_dir,
        resize_long=1024,
        gsd0=gsd_ref,
        gsd1=gsd_src,
        model=model
    )
    metrics["sift_baseline_matches"] = int(sift_matches)
    metrics["gsd_ref"] = gsd_ref
    metrics["gsd_src"] = gsd_src
    metrics["scale_ratio"] = round(float(scale_ratio), 4)

    # Save metrics JSON
    metrics_path = os.path.join(out_dir, "metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(
            metrics, f, indent=2,
            default=lambda x: x.tolist() if isinstance(x, np.ndarray) else float(x) if isinstance(x, (np.float32, np.float64)) else x
        )

    # Section 2.D Summary Table
    pair_name = f"{Path(img_low_sun).stem} <-> {Path(img_high_sun).stem}"
    print("\n" + "=" * 115)
    print(" SUMMARY METRICS TABLE - SIH26166 SCALE-INVARIANT EVALUATION")
    print("=" * 115)
    header = f"{'Pair Name':<30} | {'Scale Ratio':<11} | {'Total Matches':<13} | {'Inlier Count':<12} | {'Inlier Ratio':<12} | {'SDI':<7} | {'Checkpoint RMSE (px)':<20} | {'Checkpoint RMSE (m)':<19}"
    print(header)
    print("-" * 115)
    row = (
        f"{pair_name[:30]:<30} | "
        f"{scale_ratio:.2f}x{'':<6} | "
        f"{metrics['raw_matches']:<13} | "
        f"{metrics['inlier_count']:<12} | "
        f"{metrics['inlier_ratio']*100:.1f}%{'':<6} | "
        f"{metrics['sdi']:.3f}{'':<2} | "
        f"{metrics['checkpoint_rmse_px']:.4f} px{'':<10} | "
        f"{metrics['checkpoint_rmse_meters']:.4f} m"
    )
    print(row)
    print("=" * 115)

    print("\n[ARTIFACTS GENERATED IN]:", out_dir)
    print(" 1. match_lines.jpg          -> Side-by-side matches with inlier/outlier color lines")
    print(" 2. difference_heatmap.jpg   -> False-color error heatmap showing residual alignment")
    print(" 3. checkerboard_overlay.jpg -> Checkerboard grid proving crater rim continuity")
    print(" 4. spatial_density.jpg      -> 2D Quadtree correspondence density map")
    print(" 5. registered_target.jpg    -> Sub-pixel warped registered target image")
    print(" 6. corresponding_match_points.csv -> Dynamic tie points with residual errors")
    print(" 7. metrics.json             -> Full metrics dictionary for SIH evaluation")
    print("=" * 78 + "\n")

    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SIH26166 Lunar Image Correspondence Pipeline")
    parser.add_argument("--low", default="data/sample/nac_low.png", help="Path to low-sun / reference image")
    parser.add_argument("--high", default="data/sample/nac_high.png", help="Path to high-sun / target image")
    parser.add_argument("--out", default="outputs", help="Output directory")
    parser.add_argument("--gsd_src", type=float, default=None, help="GSD of source/high-sun image")
    parser.add_argument("--gsd_ref", type=float, default=None, help="GSD of reference/low-sun image")
    parser.add_argument("--model", type=str, default="homography", choices=["homography", "affine", "tps"], help="Model")
    args = parser.parse_args()

    run_pipeline(
        img_low_sun=args.low,
        img_high_sun=args.high,
        out_dir=args.out,
        gsd_src=args.gsd_src,
        gsd_ref=args.gsd_ref,
        model=args.model
    )

