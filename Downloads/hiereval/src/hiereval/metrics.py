"""Hierarchical evaluation metrics.

Every function takes ``y_true`` and ``y_pred`` as sequences of hierarchy paths
(see :mod:`hiereval.paths`). Confidence-aware metrics additionally take a
``confidence`` argument.
"""

from __future__ import annotations

import statistics as _stats
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

from .paths import UNK, effective_depth, lca_index, taxonomic_distance, truncate

__all__ = [
    "DEFAULT_ALPHAS",
    "DistanceStats",
    "HierarchicalPRF",
    "depth_breakdown",
    "distance_stats",
    "full_path_accuracy",
    "hcwd_curve",
    "hcwd_per_sample",
    "hcwd_score",
    "hierarchical_prf",
    "mean_taxonomic_distance",
]

DEFAULT_ALPHAS: tuple[float, ...] = (0.1, 0.3, 0.5, 0.7, 1.0)


def _check_lengths(*arrays: Sequence[Any]) -> int:
    lengths = {len(a) for a in arrays}
    if len(lengths) != 1:
        raise ValueError(f"Inputs must be the same length, got {sorted(lengths)}")
    return lengths.pop()


def _resolve_confidence(
    entry: Any, predicted: Sequence[Any], unk: Any
) -> float:
    """Pick the confidence value for one sample.

    Accepts either a single float, or a per-level sequence from which the
    value at the deepest predicted level is taken.
    """
    if isinstance(entry, (int, float)):
        value = float(entry)
    else:
        depth = effective_depth(predicted, unk)
        if depth == 0:
            return 0.0
        value = float(entry[depth - 1])
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"confidence {value} is outside [0, 1]")
    return value


# --------------------------------------------------------------------------
# Confidence-Weighted Hierarchical Distance
# --------------------------------------------------------------------------

def hcwd_per_sample(
    y_true: Sequence[Sequence[Any]],
    y_pred: Sequence[Sequence[Any]],
    confidence: Sequence[Any],
    *,
    alpha: float = 0.5,
    unk: Any = UNK,
) -> list[float]:
    r"""Per-sample Confidence-Weighted Hierarchical Distance.

    .. math::

        S_i = \frac{d_{lca} \cdot c_i}{1 + \delta_i}
              - \alpha \cdot c_i \cdot \delta_i

    where :math:`d_{lca}` is the zero-based level index of the lowest common
    ancestor, :math:`c_i` the model's confidence, and :math:`\delta_i` the
    taxonomic distance.

    **Higher is better.** Despite "distance" in the name, HCWD is a reward
    minus a penalty rather than a distance, and it can go negative.

    The first term rewards confidently sharing deep structure with the truth.
    The second penalises in proportion to *both* confidence and error size,
    encoding the principle that an assertively wrong prediction is worse than
    a hesitant one. Because the LCA index is ``-1`` when nothing is shared, a
    prediction in a wholly unrelated subtree is penalised twice over.

    Parameters
    ----------
    confidence : sequence
        Either one float per sample, or one sequence of per-level confidences
        per sample, in which case the value at the deepest predicted level is
        used.
    alpha : float, default 0.5
        Penalty weight. Sweep it with :func:`hcwd_curve` rather than trusting
        a single value.
    """
    n = _check_lengths(y_true, y_pred, confidence)
    if alpha < 0:
        raise ValueError("alpha must be non-negative")

    scores: list[float] = []
    for i in range(n):
        c = _resolve_confidence(confidence[i], y_pred[i], unk)
        delta = taxonomic_distance(y_pred[i], y_true[i], unk)
        lca = lca_index(y_pred[i], y_true[i], unk)
        scores.append((lca * c) / (1 + delta) - alpha * c * delta)
    return scores


def hcwd_score(
    y_true: Sequence[Sequence[Any]],
    y_pred: Sequence[Sequence[Any]],
    confidence: Sequence[Any],
    *,
    alpha: float = 0.5,
    unk: Any = UNK,
) -> float:
    """Mean HCWD across all samples. Higher is better.

    See :func:`hcwd_per_sample` for the definition.
    """
    scores = hcwd_per_sample(y_true, y_pred, confidence, alpha=alpha, unk=unk)
    return _stats.fmean(scores) if scores else 0.0


def hcwd_curve(
    y_true: Sequence[Sequence[Any]],
    y_pred: Sequence[Sequence[Any]],
    confidence: Sequence[Any],
    *,
    alphas: Sequence[float] = DEFAULT_ALPHAS,
    unk: Any = UNK,
) -> dict[float, dict[str, float]]:
    """HCWD across a range of penalty weights.

    A single alpha hides what matters. Sweeping it shows *how fast* a model's
    score degrades as confident errors are punished harder — a model whose
    score crosses zero is one whose confident mistakes outweigh everything it
    gets right.

    Returns
    -------
    dict
        ``{alpha: {"mean": float, "std": float}}``, in the order given.
    """
    out: dict[float, dict[str, float]] = {}
    for a in alphas:
        scores = hcwd_per_sample(y_true, y_pred, confidence, alpha=a, unk=unk)
        out[a] = {
            "mean": _stats.fmean(scores) if scores else 0.0,
            "std": _stats.pstdev(scores) if len(scores) > 1 else 0.0,
        }
    return out


# --------------------------------------------------------------------------
# Set-based metrics
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class HierarchicalPRF:
    """Set-based hierarchical precision, recall and F1."""

    precision: float
    recall: float
    f1: float
    n_samples: int = 0
    n_skipped: int = 0

    def __str__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"hP={self.precision:.4f} hR={self.recall:.4f} hF1={self.f1:.4f}"
        )


def hierarchical_prf(
    y_true: Sequence[Sequence[Any]],
    y_pred: Sequence[Sequence[Any]],
    *,
    unk: Any = UNK,
) -> HierarchicalPRF:
    """Micro-averaged set-based hierarchical precision, recall and F1.

    Each path becomes a set of ``(level, label)`` pairs with padding excluded,
    so the same label at two different levels is not treated as a match.
    Precision is the share of predicted nodes that are correct; recall is the
    share of true nodes recovered.

    Samples whose ground truth is entirely padding are skipped and counted.
    Padding in a *prediction* is not skipped: missing nodes cost recall.

    These measures reward partial correctness, but they are generous — credit
    for coarse levels can mask leaf-level failure. Report them alongside
    distance-based measures, never instead of them.
    """
    _check_lengths(y_true, y_pred)
    inter = pred_total = true_total = 0
    skipped = 0

    for t, p in zip(y_true, y_pred, strict=True):
        t_set = {(i, lbl) for i, lbl in enumerate(truncate(t, unk))}
        if not t_set:
            skipped += 1
            continue
        p_set = {(i, lbl) for i, lbl in enumerate(truncate(p, unk))}
        inter += len(t_set & p_set)
        pred_total += len(p_set)
        true_total += len(t_set)

    precision = inter / pred_total if pred_total else 0.0
    recall = inter / true_total if true_total else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall)
        else 0.0
    )
    return HierarchicalPRF(
        precision, recall, f1, len(y_true) - skipped, skipped
    )


def depth_breakdown(
    y_true: Sequence[Sequence[Any]],
    y_pred: Sequence[Sequence[Any]],
    *,
    unk: Any = UNK,
) -> dict[int, HierarchicalPRF]:
    """Hierarchical PRF grouped by the true depth of each sample.

    Aggregate scores hide the usual pattern: performance holds at shallow
    depths and collapses at the leaves.
    """
    _check_lengths(y_true, y_pred)
    groups: dict[int, tuple[list[Any], list[Any]]] = {}
    for t, p in zip(y_true, y_pred, strict=True):
        depth = effective_depth(t, unk)
        if depth == 0:
            continue
        ts, ps = groups.setdefault(depth, ([], []))
        ts.append(t)
        ps.append(p)
    return {
        d: hierarchical_prf(ts, ps, unk=unk)
        for d, (ts, ps) in sorted(groups.items())
    }


def full_path_accuracy(
    y_true: Sequence[Sequence[Any]],
    y_pred: Sequence[Sequence[Any]],
    *,
    unk: Any = UNK,
) -> float:
    """Share of samples matching the truth at every level.

    Equivalent to the proportion of samples with a taxonomic distance of zero,
    and the strictest of the hierarchical measures.
    """
    n = _check_lengths(y_true, y_pred)
    if n == 0:
        return 0.0
    exact = sum(
        taxonomic_distance(p, t, unk) == 0
        for t, p in zip(y_true, y_pred, strict=True)
    )
    return exact / n


# --------------------------------------------------------------------------
# Distance-based metrics
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class DistanceStats:
    """Summary of taxonomic distances across a prediction set."""

    micro: float
    macro: float
    std: float
    minimum: int
    maximum: int
    perfect_matches: int
    perfect_rate: float
    distribution: dict[int, int] = field(default_factory=dict)
    lca_distribution: dict[int, int] = field(default_factory=dict)


def mean_taxonomic_distance(
    y_true: Sequence[Sequence[Any]],
    y_pred: Sequence[Sequence[Any]],
    *,
    unk: Any = UNK,
    average: str = "micro",
    key: Callable[[Sequence[Any]], Any] | None = None,
) -> float:
    """Mean taxonomic distance.

    Parameters
    ----------
    average : {"micro", "macro"}
        ``micro`` weights every sample equally. ``macro`` averages within each
        class first, so rare classes count as much as common ones. The gap
        between the two reads directly as how much worse the model behaves in
        the long tail.
    key : callable, optional
        Maps a true path to its class identity for macro averaging. Defaults
        to the full truncated path.
    """
    n = _check_lengths(y_true, y_pred)
    if n == 0:
        return 0.0
    distances = [
        taxonomic_distance(p, t, unk)
        for t, p in zip(y_true, y_pred, strict=True)
    ]
    if average == "micro":
        return _stats.fmean(distances)
    if average != "macro":
        raise ValueError(f"average must be 'micro' or 'macro', got {average!r}")

    key = key or (lambda path: tuple(truncate(path, unk)))
    buckets: dict[Any, list[int]] = {}
    for t, d in zip(y_true, distances, strict=True):
        buckets.setdefault(key(t), []).append(d)
    return _stats.fmean([_stats.fmean(v) for v in buckets.values()])


def distance_stats(
    y_true: Sequence[Sequence[Any]],
    y_pred: Sequence[Sequence[Any]],
    *,
    unk: Any = UNK,
    key: Callable[[Sequence[Any]], Any] | None = None,
) -> DistanceStats:
    """Full taxonomic distance summary, including both distributions.

    The distance histogram is the most informative single view of a
    hierarchical model: a mass at distance 2 means sibling confusion, a tail
    at 6 or more means the model is losing whole subtrees.
    """
    n = _check_lengths(y_true, y_pred)
    if n == 0:
        return DistanceStats(0.0, 0.0, 0.0, 0, 0, 0, 0.0, {}, {})

    distances = [
        taxonomic_distance(p, t, unk)
        for t, p in zip(y_true, y_pred, strict=True)
    ]
    lcas = [
        lca_index(p, t, unk) for t, p in zip(y_true, y_pred, strict=True)
    ]
    perfect = sum(d == 0 for d in distances)
    return DistanceStats(
        micro=_stats.fmean(distances),
        macro=mean_taxonomic_distance(
            y_true, y_pred, unk=unk, average="macro", key=key
        ),
        std=_stats.pstdev(distances) if n > 1 else 0.0,
        minimum=min(distances),
        maximum=max(distances),
        perfect_matches=perfect,
        perfect_rate=perfect / n,
        distribution=dict(sorted(Counter(distances).items())),
        lca_distribution=dict(sorted(Counter(lcas).items())),
    )
