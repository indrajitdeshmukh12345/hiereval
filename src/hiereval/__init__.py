"""Evaluation metrics for hierarchical classification.

Flat accuracy treats every mistake as equally wrong. In a taxonomy it is not:
confusing two cat breeds is a near-miss, confusing a cat with a lorry is a
different kind of failure. And no standard metric distinguishes a model that
was 40% sure of a wrong answer from one that was 99% sure.

This package measures both: how far a prediction landed from the truth, and
how confident the model was when it got there.
"""

from .errors import (
    ErrorType,
    Taxonomy,
    classify_error,
    distance_distribution,
    error_profile,
    inconsistency_rate,
)
from .levels import LevelScores, per_level_scores
from .metrics import (
    DEFAULT_ALPHAS,
    DistanceStats,
    HierarchicalPRF,
    depth_breakdown,
    distance_stats,
    full_path_accuracy,
    hcwd_curve,
    hcwd_per_sample,
    hcwd_score,
    hierarchical_prf,
    mean_taxonomic_distance,
)
from .paths import (
    UNK,
    effective_depth,
    lca_depth,
    lca_index,
    taxonomic_distance,
    truncate,
)
from .report import Report, evaluate

__version__ = "0.1.0"

__all__ = [
    "DEFAULT_ALPHAS",
    "UNK",
    "DistanceStats",
    "ErrorType",
    "HierarchicalPRF",
    "LevelScores",
    "Report",
    "Taxonomy",
    "classify_error",
    "depth_breakdown",
    "distance_distribution",
    "distance_stats",
    "effective_depth",
    "error_profile",
    "evaluate",
    "full_path_accuracy",
    "hcwd_curve",
    "hcwd_per_sample",
    "hcwd_score",
    "hierarchical_prf",
    "inconsistency_rate",
    "lca_depth",
    "lca_index",
    "mean_taxonomic_distance",
    "per_level_scores",
    "taxonomic_distance",
    "truncate",
]
