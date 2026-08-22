"""Per-level flat metrics.

These are ordinary precision, recall and F1 computed independently at each
taxonomic level. They are included for comparability with prior work that
reports flat numbers, and because they are what most readers expect to see
first.

They are *not* sufficient on their own. A flat metric cannot distinguish a
sibling error from a jump across the taxonomy, and it cannot see confidence at
all. Report these alongside the structure-aware measures in
:mod:`hiereval.metrics`, never instead of them.
"""

from __future__ import annotations

import statistics as _stats
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from .paths import UNK

__all__ = ["LevelScores", "per_level_scores"]


@dataclass(frozen=True)
class LevelScores:
    """Flat metrics for a single taxonomic level."""

    level: int
    accuracy: float
    precision_micro: float
    recall_micro: float
    f1_micro: float
    precision_macro: float
    recall_macro: float
    f1_macro: float
    n_valid: int

    def __str__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"Level {self.level}: acc={self.accuracy:.4f} "
            f"F1_macro={self.f1_macro:.4f} (n={self.n_valid})"
        )


def _macro_prf(
    trues: Sequence[Any], preds: Sequence[Any]
) -> tuple[float, float, float]:
    """Macro-averaged precision, recall and F1 over all labels seen."""
    labels = set(trues) | set(preds)
    precisions, recalls, f1s = [], [], []
    for label in labels:
        tp = sum(t == label and p == label for t, p in zip(trues, preds, strict=True))
        fp = sum(t != label and p == label for t, p in zip(trues, preds, strict=True))
        fn = sum(t == label and p != label for t, p in zip(trues, preds, strict=True))
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall)
            else 0.0
        )
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)
    if not labels:
        return 0.0, 0.0, 0.0
    return _stats.fmean(precisions), _stats.fmean(recalls), _stats.fmean(f1s)


def per_level_scores(
    y_true: Sequence[Sequence[Any]],
    y_pred: Sequence[Sequence[Any]],
    *,
    unk: Any = UNK,
    ignore_unknown_predictions: bool = False,
) -> list[LevelScores]:
    """Flat accuracy, precision, recall and F1 at each level.

    Samples whose *true* label at a level is padding are excluded from that
    level, since there is nothing to score against.

    Parameters
    ----------
    ignore_unknown_predictions : bool, default False
        When ``False``, a padded prediction counts as wrong. When ``True``, it
        is dropped from the calculation instead, which inflates the scores but
        matches some published evaluation scripts. Leave it off unless you are
        reproducing a specific set of numbers.

    Notes
    -----
    For single-label classification, micro-averaged precision, recall and F1
    are all mathematically equal to accuracy. They are reported separately
    only because readers expect the columns.
    """
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must be the same length")

    depth = max((len(p) for p in y_true), default=0)
    results: list[LevelScores] = []

    for level in range(depth):
        trues, preds = [], []
        for t, p in zip(y_true, y_pred, strict=True):
            if level >= len(t) or t[level] == unk:
                continue
            predicted = p[level] if level < len(p) else unk
            if ignore_unknown_predictions and predicted == unk:
                continue
            trues.append(t[level])
            preds.append(predicted)

        n = len(trues)
        if n == 0:
            results.append(
                LevelScores(
                    level=level + 1,
                    accuracy=0.0,
                    precision_micro=0.0,
                    recall_micro=0.0,
                    f1_micro=0.0,
                    precision_macro=0.0,
                    recall_macro=0.0,
                    f1_macro=0.0,
                    n_valid=0,
                )
            )
            continue

        correct = sum(t == p for t, p in zip(trues, preds, strict=True))
        accuracy = correct / n
        p_macro, r_macro, f_macro = _macro_prf(trues, preds)
        results.append(
            LevelScores(
                level=level + 1,
                accuracy=accuracy,
                precision_micro=accuracy,
                recall_micro=accuracy,
                f1_micro=accuracy,
                precision_macro=p_macro,
                recall_macro=r_macro,
                f1_macro=f_macro,
                n_valid=n,
            )
        )
    return results
