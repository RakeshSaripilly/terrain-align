"""Importable package entry point for batch ISRO evaluation."""

import argparse
import importlib.util
from pathlib import Path
from types import ModuleType


_LEGACY_PATH = Path(__file__).with_name("Batch-Isro-Evaluator.py")


def _load_legacy() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "SunAngle._batch_isro_evaluator_legacy", _LEGACY_PATH
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load batch evaluator: {_LEGACY_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_implementation = _load_legacy()
find_pairs_dot_pattern = _implementation.find_pairs_dot_pattern
find_pairs_two_dirs_dot_pattern = _implementation.find_pairs_two_dirs_dot_pattern
run_batch = _implementation.run_batch

__all__ = [
    "find_pairs_dot_pattern",
    "find_pairs_two_dirs_dot_pattern",
    "run_batch",
]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Batch ISRO evaluation for <image_id>.1 and <image_id>.2 images"
    )
    parser.add_argument("--pair_dir", help="Folder containing both .1 and .2 images")
    parser.add_argument("--low_dir", help="Folder containing low-sun .2 images")
    parser.add_argument("--high_dir", help="Folder containing high-sun .1 images")
    parser.add_argument("--out", default="batch_outputs_quickmap_dot")
    parser.add_argument("--resize", type=int, default=1024)
    parser.add_argument("--conf", type=float, default=0.20)
    parser.add_argument("--max_pairs", type=int, default=None)
    parser.add_argument("--no_vis", action="store_true")
    parser.add_argument("--gsd_src", type=float, default=None, help="GSD source/high-sun")
    parser.add_argument("--gsd_ref", type=float, default=None, help="GSD reference/low-sun")
    parser.add_argument("--model", type=str, default="homography", choices=["homography", "affine", "tps"])
    args = parser.parse_args()

    if not args.pair_dir and not (args.low_dir and args.high_dir):
        parser.error("provide --pair_dir or both --low_dir and --high_dir")

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
        model=args.model,
    )



if __name__ == "__main__":
    main()
