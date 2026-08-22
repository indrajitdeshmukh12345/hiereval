"""Hierarchy paths, depths and taxonomic distance.

A *path* is a sequence of labels from the root downwards, one per taxonomic
level, e.g. ``["Animal", "Cat", "Siamese", "UNK"]``.

Real taxonomies are ragged: some branches run four levels deep, others stop at
two. Paths are therefore padded to a fixed length with an *unknown token*, and
that padding must be excluded from every calculation.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

__all__ = [
    "UNK",
    "effective_depth",
    "truncate",
    "lca_index",
    "lca_depth",
    "taxonomic_distance",
]

UNK = "UNK"
"""Default token marking a missing label."""


def effective_depth(path: Sequence[Any], unk: Any = UNK) -> int:
    """Number of real labels at the head of ``path``.

    Only the leading run counts. ``["Animal", "UNK", "Siamese"]`` has an
    effective depth of 1: once the chain to the root is broken, deeper labels
    have no verifiable position in the taxonomy.
    """
    depth = 0
    for label in path:
        if label == unk:
            break
        depth += 1
    return depth


def truncate(path: Sequence[Any], unk: Any = UNK) -> list[Any]:
    """Return ``path`` with padding stripped."""
    return list(path[: effective_depth(path, unk)])


def lca_index(
    predicted: Sequence[Any], true: Sequence[Any], unk: Any = UNK
) -> int:
    """Zero-based level index of the lowest common ancestor.

    Because every node in a tree has a unique root path, the LCA is the
    longest shared prefix. Returns ``-1`` when the paths share nothing at all,
    not even the first level.

    This index convention — rather than a count of shared levels — is what
    :func:`~hiereval.metrics.hcwd_score` uses, so that a prediction with no
    common ancestor is penalised rather than merely unrewarded.
    """
    p, t = truncate(predicted, unk), truncate(true, unk)
    index = -1
    for level, (a, b) in enumerate(zip(p, t, strict=False)):
        if a != b:
            break
        index = level
    return index


def lca_depth(
    predicted: Sequence[Any], true: Sequence[Any], unk: Any = UNK
) -> int:
    """Number of levels shared by two paths. Zero when nothing is shared.

    This is ``lca_index() + 1``, provided for readability where a count is
    more natural than an index.
    """
    return lca_index(predicted, true, unk) + 1


def taxonomic_distance(
    predicted: Sequence[Any], true: Sequence[Any], unk: Any = UNK
) -> int:
    """Edge-counting distance between a predicted and a true path.

    The number of edges walked up from the prediction to the lowest common
    ancestor, plus the number walked back down to the truth.

    ``0`` is an exact hierarchical match. ``1`` is one level of over- or
    under-specialisation, ``2`` a typical sibling error, and values of ``6``
    or more usually mean the prediction landed in a different subtree.
    """
    lca = lca_index(predicted, true, unk)
    pred_leaf = effective_depth(predicted, unk) - 1
    true_leaf = effective_depth(true, unk) - 1
    return (pred_leaf - lca) + (true_leaf - lca)
