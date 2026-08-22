"""A single call that runs the whole evaluation framework."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from .errors import Taxonomy, error_profile, inconsistency_rate
from .levels import LevelScores, per_level_scores
from .metrics import (
    DEFAULT_ALPHAS,
    DistanceStats,
    HierarchicalPRF,
    depth_breakdown,
    distance_stats,
    full_path_accuracy,
    hcwd_curve,
    hierarchical_prf,
)
from .paths import UNK

__all__ = ["Report", "evaluate"]


@dataclass
class Report:
    """Everything the framework measures, for one model."""

    name: str
    n_samples: int
    per_level: list[LevelScores]
    hierarchical: HierarchicalPRF
    by_depth: dict[int, HierarchicalPRF]
    full_path_accuracy: float
    distance: DistanceStats
    errors: dict[str, float]
    hcwd: dict[float, dict[str, float]] = field(default_factory=dict)
    inconsistency_rate: float | None = None

    def summary(self) -> str:
        """A readable block suitable for pasting into a results table."""
        lines = [
            f"=== {self.name} ({self.n_samples} samples) ===",
            "",
            "Per-level (flat):",
        ]
        for level in self.per_level:
            lines.append(
                f"  L{level.level}: acc={level.accuracy:.4f}  "
                f"F1_macro={level.f1_macro:.4f}  n={level.n_valid}"
            )
        lines += [
            "",
            "Hierarchical (set-based):",
            f"  hP={self.hierarchical.precision:.4f}  "
            f"hR={self.hierarchical.recall:.4f}  "
            f"hF1={self.hierarchical.f1:.4f}",
        ]
        for depth, prf in self.by_depth.items():
            lines.append(f"  depth {depth}: hF1={prf.f1:.4f}")
        lines += [
            "",
            f"Full-path accuracy: {self.full_path_accuracy:.4f}",
            "",
            "Taxonomic distance:",
            f"  micro={self.distance.micro:.4f} "
            f"(sd {self.distance.std:.4f})  macro={self.distance.macro:.4f}",
            f"  exact matches: {self.distance.perfect_matches} "
            f"({self.distance.perfect_rate:.2%})",
            f"  distribution: {self.distance.distribution}",
            "",
            "Error profile:",
        ]
        for kind, share in self.errors.items():
            lines.append(f"  {kind:<16} {share:.2%}")
        if self.hcwd:
            lines += ["", "HCWD (higher is better):"]
            for alpha, stats in self.hcwd.items():
                lines.append(
                    f"  alpha={alpha:<4} mean={stats['mean']:+.4f} "
                    f"sd={stats['std']:.4f}"
                )
        if self.inconsistency_rate is not None:
            lines += [
                "",
                f"Inconsistency rate (TICE): {self.inconsistency_rate:.2%}",
            ]
        return "\n".join(lines)


def evaluate(
    y_true: Sequence[Sequence[Any]],
    y_pred: Sequence[Sequence[Any]],
    confidence: Sequence[Any] | None = None,
    *,
    name: str = "model",
    unk: Any = UNK,
    alphas: Sequence[float] = DEFAULT_ALPHAS,
    taxonomy: Taxonomy | None = None,
    severe_distance: int = 6,
) -> Report:
    """Run every metric in the package and return one :class:`Report`.

    Parameters
    ----------
    confidence : sequence, optional
        One float per sample, or one per-level sequence per sample. Omit it to
        skip HCWD; every other measure still runs.
    taxonomy : Taxonomy, optional
        Supply one built from the ground-truth paths to also compute the
        inconsistency rate (TICE).

    Examples
    --------
    >>> report = evaluate(y_true, y_pred, confidences, name="multi-head")
    >>> print(report.summary())
    """
    return Report(
        name=name,
        n_samples=len(y_true),
        per_level=per_level_scores(y_true, y_pred, unk=unk),
        hierarchical=hierarchical_prf(y_true, y_pred, unk=unk),
        by_depth=depth_breakdown(y_true, y_pred, unk=unk),
        full_path_accuracy=full_path_accuracy(y_true, y_pred, unk=unk),
        distance=distance_stats(y_true, y_pred, unk=unk),
        errors=error_profile(
            y_true, y_pred, unk=unk, severe_distance=severe_distance
        ),
        hcwd=(
            hcwd_curve(y_true, y_pred, confidence, alphas=alphas, unk=unk)
            if confidence is not None
            else {}
        ),
        inconsistency_rate=(
            inconsistency_rate(y_pred, taxonomy, unk=unk)
            if taxonomy is not None
            else None
        ),
    )
