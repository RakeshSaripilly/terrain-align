"""Importable package entry point for the ISRO Task 2 evaluator.

The historical implementation retains its hyphenated script filename for
backwards-compatible command-line use. This module provides the normal Python
module name used by the package and by ``task2pipeline``.
"""

import importlib.util
from pathlib import Path
from types import ModuleType


_LEGACY_PATH = Path(__file__).with_name("Isro-Metric-Evaluator.py")


def _load_legacy() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "SunAngle._isro_metric_evaluator_legacy", _LEGACY_PATH
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load evaluator implementation: {_LEGACY_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_implementation = _load_legacy()
compute_reprojection_errors = _implementation.compute_reprojection_errors
compute_isro_metrics = _implementation.compute_isro_metrics
save_match_points_csv = _implementation.save_match_points_csv
save_isro_deliverables = _implementation.save_isro_deliverables

__all__ = [
    "compute_reprojection_errors",
    "compute_isro_metrics",
    "save_match_points_csv",
    "save_isro_deliverables",
]
