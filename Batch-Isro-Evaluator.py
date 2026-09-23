import os
import sys
import re
import glob
import json
import csv
import time
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import cv2
import numpy as np

# Safe UTF-8 console output for Windows cp1252
if sys.stdout is not None and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

try:
    from .pipeline import LunarCorrespondenceEngine
    from .isro_metric_evaluator import save_isro_deliverables
    from .evaluation import sift_baseline
    from .preprocessing import infer_gsd_from_path
except (ImportError, ValueError):
    from pipeline import LunarCorrespondenceEngine
    from isro_metric_evaluator import save_isro_deliverables
    from evaluation import sift_baseline
    from preprocessing import infer_gsd_from_path


def find_pairs_dot_pattern(pair_dir: str) -> List[Tuple[str, str, str]]:

    """
    Your pattern: <img_no>.1.png = high, <img_no>.2.png = low
    Example: 1.1.png + 1.2.png, 23.1.png + 23.2.png
    Also handles: 001.1.png + 001.2.png
    """
    pair_dir = Path(pair_dir)
    # Regex: captures img_no and illumination id (1 or 2)
    # Handles: 1.1.png, 001.1.png, 100.2.png, crater_5.1.png -> tricky, so use last dot split
    pattern = re.compile(r"^(.*)\.([12])\.(png|jpg|jpeg|tif|tiff)$", re.IGNORECASE)
    
    groups = {}  # img_no -> { '1': path, '2': path }
    
    all_files = list(pair_dir.glob("*.png")) + list(pair_dir.glob("*.jpg")) + list(pair_dir.glob("*.jpeg")) + list(pair_dir.glob("*.PNG")) + list(pair_dir.glob("*.JPG"))
    
    for p in all_files:
        m = pattern.match(p.name)
        if m:
            base = m.group(1)  # img_no
            illum = m.group(2)  # 1 or 2
            if base not in groups:
                groups[base] = {}
            groups[base][illum] = p
        else:
            # Also try pattern like 1.1.png where stem is "1.1" - split by .
            # Some OS saves as "1.1.png" -> stem "1.1" -> split
            stem = p.stem  # "1.1" for "1.1.png"
            if "." in stem:
                parts = stem.rsplit(".", 1)
                if parts[1] in ["1","2"] and parts[0]:
                    base = parts[0]
                    illum = parts[1]
                    if base not in groups:
                        groups[base] = {}
                    groups[base][illum] = p
    
    pairs = []
    for base, illum_map in groups.items():
        if "1" in illum_map and "2" in illum_map:
            # You said: .1 = low (less visibility), .2 = high (visible crater)
            low_path = illum_map["1"]  # visible
            high_path = illum_map["2"]   # less visible
            # For pipeline, we use low as reference? But either works - keep consistent
            # We'll treat .2 (low) as reference, .1 (high) as target to show improvement
            pairs.append((base, str(low_path), str(high_path)))
    
    # Sort numerically by img_no if possible
    def sort_key(x):
        try:
            return int(x[0])
        except:
            try:
                return int(re.search(r"\d+", x[0]).group())
            except:
                return x[0]
    
    pairs = sorted(pairs, key=sort_key)
    return pairs


def find_pairs_two_dirs_dot_pattern(low_dir: str, high_dir: str) -> List[Tuple[str, str, str]]:
    """If you split .1 and .2 into separate folders"""
    # Actually if you have low_dir with *.2.png and high_dir with *.1.png
    low_pairs = find_pairs_dot_pattern(low_dir)  # will find only half, so handle separately
    high_pairs = find_pairs_dot_pattern(high_dir)
    
    # Better: just list all files in each dir and match by base number
    low_dir_p = Path(low_dir)
    high_dir_p = Path(high_dir)
    
    low_files = list(low_dir_p.glob("*.png")) + list(low_dir_p.glob("*.jpg"))
    high_files = list(high_dir_p.glob("*.png")) + list(high_dir_p.glob("*.jpg"))
    
    # Extract base number
    def extract_base(p: Path):
        name = p.name
        # Remove .1.png or .2.png suffix
        m = re.match(r"^(.*)\.[12]\.(?:png|jpg|jpeg)$", name, re.IGNORECASE)
        if m:
            return m.group(1)
        # Try stem split
        stem = p.stem
        if "." in stem:
            return stem.rsplit(".",1)[0]
        return stem
    
    low_map = {extract_base(p): p for p in low_files}
    high_map = {extract_base(p): p for p in high_files}
    
    common = set(low_map.keys()) & set(high_map.keys())
    pairs = []
    for base in sorted(common, key=lambda x: int(x) if x.isdigit() else x):
        pairs.append((base, str(low_map[base]), str(high_map[base])))
    
    # If no common due to .1/.2 being separated (low has only .2, high only .1), pair by base
    if len(pairs) == 0:
        # Assume low_dir contains *.2.png, high_dir contains *.1.png, same base numbers
        # Rebuild with just numeric base
        low_bases = {}
        for p in low_files:
            b = extract_base(p)
            low_bases[b] = p
        high_bases = {}
        for p in high_files:
            b = extract_base(p)
            high_bases[b] = p
        common = set(low_bases.keys()) & set(high_bases.keys())
        for base in sorted(common, key=lambda x: int(x) if x.isdigit() else x):
            pairs.append((base, str(low_bases[base]), str(high_bases[base])))
    
    return pairs


def run_batch(
    low_dir: str = None,
    high_dir: str = None,
    pair_dir: str = None,
    out_dir: str = "batch_outputs_quickmap",
    resize_long: int = 1024,
    conf_thresh: float = 0.20,
    ransac_thresh: float = 2.0,
    save_vis: bool = True,
    max_pairs: int = None,
    gsd_src: Optional[float] = None,
    gsd_ref: Optional[float] = None,
    model: str = "homography"
):
    # Find pairs with your .1/.2 pattern first
    if pair_dir:
        print(f"[BATCH] Single-folder mode with .1/.2 pattern: {pair_dir}")
        pairs = find_pairs_dot_pattern(pair_dir)
        if len(pairs) == 0:
            from pathlib import Path as P
            all_png = list(P(pair_dir).glob("*.png"))
            print(f"  Found {len(all_png)} files but no .1/.2 pattern, trying generic")
            pairs = []
    elif low_dir and high_dir:
        print(f"[BATCH] Two-dir mode: {low_dir} + {high_dir}")
        pairs = find_pairs_two_dirs_dot_pattern(low_dir, high_dir)
    else:
        print("Provide --pair_dir OR --low_dir + --high_dir")
        return

    if len(pairs) == 0:
        print("[ERROR] No pairs found with pattern <img_no>.1.png + <img_no>.2.png")
        print("  Example expected: 1.1.png + 1.2.png, 2.1.png + 2.2.png")
        print(f"  Checked dir(s): {pair_dir or low_dir}")
        if pair_dir:
            sample = list(Path(pair_dir).glob("*"))[:10]
            print(f"  Sample files in dir: {[p.name for p in sample]}")
        return

    if max_pairs:
        pairs = pairs[:max_pairs]

    print(f"[BATCH] Found {len(pairs)} pairs (200 files = 100 pairs expected)")
    for i, (pid, low, high) in enumerate(pairs[:5]):
        print(f"  {i+1}. ID={pid}: low={Path(low).name} (less visible .2) <-> high={Path(high).name} (visible .1)")
    if len(pairs) > 5:
        print(f"  ... and {len(pairs)-5} more")

    print("\n[BATCH] Loading LoFTR outdoor (once)...")
    engine = LunarCorrespondenceEngine()
    print(f"  Device: {engine.matcher.device}")

    all_metrics = []
    failed = []
    t0 = time.time()

    for idx, (pair_id, low_path, high_path) in enumerate(pairs):
        print(f"\n{'='*80}")
        print(f" [{idx+1}/{len(pairs)}] Pair {pair_id}: {Path(low_path).name} <-> {Path(high_path).name}")
        pair_out = os.path.join(out_dir, f"{int(pair_id):03d}_{pair_id}" if str(pair_id).isdigit() else f"{idx:03d}_{pair_id}")
        os.makedirs(pair_out, exist_ok=True)

        pair_gsd_ref = gsd_ref if gsd_ref is not None else infer_gsd_from_path(low_path)
        pair_gsd_src = gsd_src if gsd_src is not None else infer_gsd_from_path(high_path)

        try:
            try:
                sift_n = sift_baseline(low_path, high_path)
            except Exception:
                sift_n = 0

            result = engine.match(
                low_path, high_path,
                resize_long=resize_long,
                conf_thresh=conf_thresh,
                subpixel_refinement=True,
                enforce_uniformity=True,
                max_uniform_points=1200,
                ransac_thresh=ransac_thresh,
                gsd0=pair_gsd_ref,
                gsd1=pair_gsd_src,
                model=model
            )

            metrics = save_isro_deliverables(
                result, out_dir=pair_out, save_registered=save_vis,
                gsd_ref=pair_gsd_ref, gsd_src=pair_gsd_src
            )
            metrics["pair_id"] = pair_id
            metrics["low_path"] = low_path
            metrics["high_path"] = high_path
            metrics["sift_baseline"] = sift_n
            all_metrics.append(metrics)

            status = "[PASS]" if metrics["isro_pass"] else "[FAIL]"
            print(f"  -> {status} RMSE={metrics['rmse_pixels']:.3f}px SDI={metrics.get('sdi', 0.0):.3f} Cov3x3={metrics['uniformity_coverage_3x3']*100:.0f}% Inliers={metrics['inlier_count']} SIFT={sift_n}")

            if not metrics["isro_pass"]:
                failed.append((pair_id, metrics["rmse_pixels"], metrics["uniformity_coverage_3x3"]))

        except Exception as e:
            print(f"  -> ERROR: {e}")
            import traceback; traceback.print_exc()
            all_metrics.append({
                "pair_id": pair_id,
                "low_path": low_path,
                "high_path": high_path,
                "total_matches": 0,
                "inlier_count": 0,
                "inlier_ratio": 0.0,
                "rmse_pixels": 999.0,
                "checkpoint_rmse_px": 999.0,
                "checkpoint_rmse_meters": 999.0,
                "sdi": 0.0,
                "uniformity_coverage_3x3": 0,
                "isro_pass": False,
                "error": str(e)
            })
            failed.append((pair_id, 999.0, 0.0))

    batch_time = time.time() - t0

    if len(all_metrics) > 0:
        rmses = [m["rmse_pixels"] for m in all_metrics if m["rmse_pixels"] < 100]
        chk_rmses = [m.get("checkpoint_rmse_px", 999.0) for m in all_metrics if m.get("checkpoint_rmse_px", 999.0) < 100]
        sdis = [m.get("sdi", 0.0) for m in all_metrics]
        inlier_ratios = [m.get("inlier_ratio", 0.0) for m in all_metrics]
        covs = [m.get("uniformity_coverage_3x3", 0) for m in all_metrics]
        passed = sum(1 for m in all_metrics if m.get("isro_pass", False))

        summary = {
            "total_pairs": len(pairs),
            "evaluated": len(all_metrics),
            "isro_pass_count": int(passed),
            "isro_pass_rate": round(passed / len(all_metrics) * 100, 1),
            "mean_rmse": round(float(np.mean(rmses)), 4) if rmses else 999,
            "median_rmse": round(float(np.median(rmses)), 4) if rmses else 999,
            "mean_checkpoint_rmse_px": round(float(np.mean(chk_rmses)), 4) if chk_rmses else 999,
            "mean_sdi": round(float(np.mean(sdis)), 4) if sdis else 0.0,
            "mean_inlier_ratio": round(float(np.mean(inlier_ratios)) * 100, 1) if inlier_ratios else 0,
            "mean_coverage_3x3": round(float(np.mean(covs)) * 100, 1) if covs else 0,
            "batch_time_sec": round(batch_time, 1),
            "time_per_pair": round(batch_time / len(pairs), 2) if pairs else 0,
        }

        # Clean Section 2.D Summary Table
        print("\n" + "=" * 125)
        print(" BATCH SCALE-INVARIANT SUMMARY TABLE - SIH26166")
        print("=" * 125)
        table_header = f"{'Pair Name':<28} | {'Scale Ratio':<11} | {'Total Matches':<13} | {'Inlier Count':<12} | {'Inlier Ratio':<12} | {'SDI':<6} | {'Checkpoint RMSE (px)':<20} | {'Checkpoint RMSE (m)':<19}"
        print(table_header)
        print("-" * 125)
        for m in all_metrics:
            p_name = str(m.get("pair_id", "pair"))
            s_ratio = m.get("scale_ratio", 1.0)
            tot = m.get("total_matches", 0)
            inl = m.get("inlier_count", 0)
            iratio = m.get("inlier_ratio", 0.0) * 100
            sdi_v = m.get("sdi", 0.0)
            chk_px = m.get("checkpoint_rmse_px", 999.0)
            chk_m = m.get("checkpoint_rmse_meters", 999.0)
            chk_px_str = f"{chk_px:.4f} px" if chk_px < 900 else "N/A"
            chk_m_str = f"{chk_m:.4f} m" if chk_m < 900 else "N/A"
            row = f"{p_name[:28]:<28} | {s_ratio:.2f}x{'':<6} | {tot:<13} | {inl:<12} | {iratio:.1f}%{'':<6} | {sdi_v:.3f} | {chk_px_str:<20} | {chk_m_str:<19}"
            print(row)
        print("=" * 125)

        print(f" Total Pairs         : {summary['total_pairs']}")
        print(f" ISRO PASS           : {summary['isro_pass_count']}/{summary['total_pairs']} ({summary['isro_pass_rate']}%)")
        print(f" Mean RMSE           : {summary['mean_rmse']:.4f} px (target <0.5, pass <2.0)")
        print(f" Mean Checkpoint RMSE: {summary['mean_checkpoint_rmse_px']:.4f} px")
        print(f" Mean SDI            : {summary['mean_sdi']:.4f} (target >= 0.65)")
        print(f" Mean Inlier Ratio   : {summary['mean_inlier_ratio']:.1f}%")
        print(f" Mean Coverage 3x3   : {summary['mean_coverage_3x3']:.1f}% (target >77%)")

        csv_path = os.path.join(out_dir, "batch_metrics.csv")
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            fieldnames = [
                "pair_id", "total_matches", "inlier_count", "inlier_ratio",
                "inlier_count_2px", "inlier_ratio_2px", "rmse_pixels",
                "checkpoint_rmse_px", "checkpoint_rmse_meters", "sdi",
                "uniformity_coverage_3x3", "uniformity_entropy_3x3",
                "scale_ratio", "isro_pass", "sift_baseline"
            ]
            fieldnames = [k for k in fieldnames if any(k in m for m in all_metrics)]
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
            writer.writeheader()
            for m in all_metrics:
                writer.writerow(m)

        with open(os.path.join(out_dir, "batch_summary.json"), 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2)

        print(f"[Saved] {csv_path}")
        print(f"[Saved] {out_dir}/batch_summary.json")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Batch for <img_no>.1.png (high) + <img_no>.2.png (low)")
    parser.add_argument("--pair_dir", type=str, help="Folder with 1.1.png,1.2.png,2.1.png,2.2.png...")
    parser.add_argument("--low_dir", type=str, help="Folder with *.2.png")
    parser.add_argument("--high_dir", type=str, help="Folder with *.1.png")
    parser.add_argument("--out", type=str, default="batch_outputs_quickmap_dot")
    parser.add_argument("--resize", type=int, default=1024)
    parser.add_argument("--conf", type=float, default=0.20)
    parser.add_argument("--ransac", type=float, default=2.0)
    parser.add_argument("--max_pairs", type=int, default=None)
    parser.add_argument("--no_vis", action="store_true")
    parser.add_argument("--gsd_src", type=float, default=None, help="GSD source/high-sun")
    parser.add_argument("--gsd_ref", type=float, default=None, help="GSD reference/low-sun")
    parser.add_argument("--model", type=str, default="homography", choices=["homography", "affine", "tps"])
    args = parser.parse_args()

    if not args.pair_dir and not args.low_dir:
        print("Examples for your naming:")
        print("  python Batch-Isro-Evaluator.py --pair_dir data/quickmap --out batch_outputs")
        print("  python Batch-Isro-Evaluator.py --low_dir data/low_2 --high_dir data/high_1 --out batch_outputs")
        sys.exit(0)

    run_batch(
        low_dir=args.low_dir,
        high_dir=args.high_dir,
        pair_dir=args.pair_dir,
        out_dir=args.out,
        resize_long=args.resize,
        conf_thresh=args.conf,
        ransac_thresh=args.ransac,
        save_vis=not args.no_vis,
        max_pairs=args.max_pairs,
        gsd_src=args.gsd_src,
        gsd_ref=args.gsd_ref,
        model=args.model
    )

