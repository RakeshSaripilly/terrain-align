"""Lunar image correspondence pipeline package."""

from .pipeline import LunarCorrespondenceEngine
from .Task2pipeline import run_task2
from .isro_metric_evaluator import compute_isro_metrics, save_isro_deliverables


def run_batch(*args, **kwargs):
	from .batch_isro_evaluator import run_batch as batch_runner
	return batch_runner(*args, **kwargs)


def find_pairs_dot_pattern(*args, **kwargs):
	from .batch_isro_evaluator import find_pairs_dot_pattern as pair_finder
	return pair_finder(*args, **kwargs)

__all__ = [
	"LunarCorrespondenceEngine",
	"run_task2",
	"compute_isro_metrics",
	"save_isro_deliverables",
	"run_batch",
	"find_pairs_dot_pattern",
]
