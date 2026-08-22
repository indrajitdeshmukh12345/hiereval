"""Structured characterisation of hierarchical errors.

Aggregate scores tell you *how much* a model is wrong. This module tells you
*how* it is wrong, which is usually the more actionable question.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Sequence
from typing import Any

from .paths import UNK, taxonomic_distance, truncate

__all__ = [
    "ErrorType",
    "classify_error",
    "error_profile",
    "distance_distribution",
    "Taxonomy",
    "inconsistency_rate",
]


class ErrorType(str):
    """String enum of hierarchical error categories."""

    EXACT = "exact"
    GENERALISATION = "generalisation"
    SPECIALISATION = "specialisation"
    SIBLING = "sibling"
    CROSS_BRANCH = "cross_branch"
    CATASTROPHIC = "catastrophic"


def classify_error(
    true: Sequence[Any],
    predicted: Sequence[Any],
    *,
    unk: Any = UNK,
    severe_distance: int = 6,
) -> str:
    """Categorise a single prediction.

    Categories
    ----------
    exact
        Identical paths.
    generalisation
        The prediction is a proper prefix of the truth: correct but stopped
        short. Often the safest failure mode.
    specialisation
        The truth is a proper prefix of the prediction: the model went deeper
        than the annotation. In biology this can indicate an unlabelled
        subtype rather than a mistake.
    sibling
        Divergence only at the deepest level, under a shared parent. The
        classic near-miss.
    cross_branch
        Divergence part-way down the taxonomy.
    catastrophic
        Divergence at the first level, or a distance of at least
        ``severe_distance``. The prediction is in a different part of the tree
        entirely.
    """
    t, p = truncate(true, unk), truncate(predicted, unk)

    if t == p:
        return ErrorType.EXACT

    delta = taxonomic_distance(p, t, unk)
    shared = 0
    for a, b in zip(t, p, strict=False):
        if a != b:
            break
        shared += 1

    if shared == 0:
        return ErrorType.CATASTROPHIC
    if delta >= severe_distance:
        return ErrorType.CATASTROPHIC
    if p == t[: len(p)]:
        return ErrorType.GENERALISATION
    if t == p[: len(t)]:
        return ErrorType.SPECIALISATION
    if shared == len(t) - 1 and len(t) == len(p):
        return ErrorType.SIBLING
    return ErrorType.CROSS_BRANCH


def error_profile(
    y_true: Sequence[Sequence[Any]],
    y_pred: Sequence[Sequence[Any]],
    *,
    unk: Any = UNK,
    severe_distance: int = 6,
    normalise: bool = True,
) -> dict[str, float]:
    """Distribution of error categories across a prediction set."""
    counts = Counter(
        classify_error(t, p, unk=unk, severe_distance=severe_distance)
        for t, p in zip(y_true, y_pred, strict=True)
    )
    n = len(y_true)
    if not n:
        return {}
    if normalise:
        return {k: v / n for k, v in sorted(counts.items())}
    return dict(sorted(counts.items()))


def distance_distribution(
    y_true: Sequence[Sequence[Any]],
    y_pred: Sequence[Sequence[Any]],
    *,
    unk: Any = UNK,
    normalise: bool = True,
) -> dict[int, float]:
    """Histogram of taxonomic distances, keyed by edge count."""
    counts = Counter(
        taxonomic_distance(p, t, unk) for t, p in zip(y_true, y_pred, strict=True)
    )
    n = len(y_true)
    if not n:
        return {}
    if normalise:
        return {k: v / n for k, v in sorted(counts.items())}
    return dict(sorted(counts.items()))


class Taxonomy:
    """The set of parent-child edges observed in a label set.

    Built from ground-truth paths, this is what lets you tell whether a
    *prediction* is structurally possible — a question no accuracy-style
    metric can answer.
    """

    def __init__(self, edges: set[tuple[int, Any, Any]]) -> None:
        self._edges = edges

    @classmethod
    def from_paths(
        cls, paths: Iterable[Sequence[Any]], *, unk: Any = UNK
    ) -> Taxonomy:
        edges: set[tuple[int, Any, Any]] = set()
        for path in paths:
            clean = truncate(path, unk)
            pairs = zip(clean, clean[1:], strict=False)
            for level, (parent, child) in enumerate(pairs):
                edges.add((level, parent, child))
        return cls(edges)

    def is_consistent(self, path: Sequence[Any], *, unk: Any = UNK) -> bool:
        """Whether every parent-child step in ``path`` exists in the taxonomy."""
        clean = truncate(path, unk)
        return all(
            (level, parent, child) in self._edges
            for level, (parent, child) in enumerate(
                zip(clean, clean[1:], strict=False)
            )
        )

    def __len__(self) -> int:
        return len(self._edges)


def inconsistency_rate(
    y_pred: Sequence[Sequence[Any]],
    taxonomy: Taxonomy,
    *,
    unk: Any = UNK,
) -> float:
    """Tree-based Inconsistency Error (TICE): share of structurally impossible paths.

    A multi-head model with independent output heads can predict a child that
    is not a descendant of its own predicted parent — "Dog" at one level and
    "Siamese Cat" at the next. Such a prediction may still score well on
    accuracy and on set-based metrics while being taxonomically meaningless.

    Sequential models are immune by construction, since traversal guarantees a
    valid path. For global models this is the measure that shows whether a
    consistency loss actually worked.
    """
    if not len(y_pred):
        return 0.0
    bad = sum(not taxonomy.is_consistent(p, unk=unk) for p in y_pred)
    return bad / len(y_pred)
