# hiereval

**Evaluation metrics for hierarchical classification — structure-aware and confidence-aware.**

[![CI](https://github.com/indrajitdeshmukh12345/hiereval/actions/workflows/ci.yml/badge.svg)](https://github.com/indrajitdeshmukh12345/hiereval/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/hiereval.svg)](https://pypi.org/project/hiereval/)
[![Python](https://img.shields.io/pypi/pyversions/hiereval.svg)](https://pypi.org/project/hiereval/)
![License](https://img.shields.io/badge/license-MIT-blue.svg)

Zero dependencies. Fully typed.

```bash
pip install hiereval
```

## The problem

Accuracy treats every mistake as equally wrong. In a taxonomy it isn't.

Confusing a Siamese cat with a Bombay cat is a near-miss — the model found the
right family and stumbled at the leaf. Confusing a Siamese cat with a rose is a
different kind of failure. Flat metrics score both as `0`.

There is a second blind spot that even hierarchical metrics share. A model that
is 40% confident in a wrong answer and a model that is 99% confident in the
*same* wrong answer receive identical scores. In deployment they are not
remotely the same system: the hesitant one can be routed to human review, while
the assertive one fails silently.

`hiereval` measures both — how far the prediction landed from the truth, and how
sure the model was when it got there.

## Quickstart

Predictions are paths through the taxonomy, root-first, padded with `"UNK"`
where a label is unavailable.

```python
from hiereval import evaluate

y_true = [["Animal", "Cat", "Siamese", "UNK"], ...]
y_pred = [["Animal", "Cat", "Bombay",  "UNK"], ...]
confidence = [0.96, ...]          # or per-level: [[0.99, 0.97, 0.96, 0.10], ...]

report = evaluate(y_true, y_pred, confidence, name="multi-head")
print(report.summary())
```

One call runs the whole framework. Individual metrics are importable on their
own if you only want one.

## Why this exists

Two models, same dataset, same 70% full-path accuracy:

| | hierarchy-aware | flat-style |
| --- | --- | --- |
| Full-path accuracy | 0.70 | 0.70 |
| Hierarchical F1 | 0.90 | 0.70 |
| Mean taxonomic distance | 0.60 | 1.80 |
| Distance histogram | `{0: 7, 2: 3}` | `{0: 7, 6: 3}` |
| HCWD at α = 0.1 | **+1.38** | **+1.13** |
| HCWD at α = 1.0 | **+0.86** | **−0.43** |

Both get the same number of answers exactly right. But one misses by a sibling
and the other jumps to a different subtree — confidently. Only the last row
makes that visible, and a negative score is an unambiguous verdict: this model's
confident errors outweigh everything it gets right.

## Confidence-Weighted Hierarchical Distance

For each sample:

```
        d_lca · c
S  =  ───────────  −  α · c · δ
        1 + δ
```

| Symbol | Meaning |
| --- | --- |
| `d_lca` | Level index of the lowest common ancestor (`-1` when nothing is shared) |
| `c` | Model confidence at the deepest predicted level |
| `δ` | Taxonomic distance: edges up to the LCA, then back down |
| `α` | Penalty weight |

The first term rewards confidently sharing deep structure with the truth. The
second penalises in proportion to both confidence and error size. Because
`d_lca` is `-1` when the paths share nothing at all, a prediction in a wholly
unrelated subtree is penalised twice over.

**Higher is better.** Despite "distance" in the name, HCWD is a reward minus a
penalty rather than a distance, and it can go negative — which is the point.

Use `hcwd_curve` rather than a single α. The *slope* carries the information: a
flat curve means the model makes cautious mistakes, a diving curve means
assertive ones.

## What's included

| Function | What it answers |
| --- | --- |
| `evaluate` | Run everything and return one report |
| `per_level_scores` | Flat accuracy, precision, recall, F1 at each level |
| `hierarchical_prf` | How much of the correct path was recovered? |
| `depth_breakdown` | Where in the tree does performance fall apart? |
| `full_path_accuracy` | How often is every level right at once? |
| `taxonomic_distance`, `distance_stats` | How far off structurally, and how is that spread? |
| `mean_taxonomic_distance` | Micro and macro averages — the gap reads as long-tail weakness |
| `hcwd_score`, `hcwd_curve` | How costly are the errors, given how sure the model was? |
| `error_profile`, `classify_error` | *How* is it wrong — near-miss, over-specialised, or lost? |
| `inconsistency_rate` | Does it predict paths the taxonomy forbids? |

`inconsistency_rate` implements the Tree-based Inconsistency Error (TICE). A
model with independent output heads can predict `Dog` at one level and
`Siamese Cat` at the next — a path that cannot exist. Such predictions can score
well on every other measure here while being taxonomically meaningless.
Sequential models are immune by construction; for global models this is the
measure that shows whether a consistency loss actually worked.

## Variable depth

Real taxonomies are ragged: some branches run four levels deep, others stop at
two. Padding is excluded from every calculation, so partially-annotated samples
are neither rewarded nor punished for levels that were never labelled.

Padding counts from the first gap onwards. `["Animal", "UNK", "Siamese"]` has an
effective depth of 1 — once the chain to the root is broken, deeper labels have
no verifiable position in the tree.

Any sentinel works via `unk=`; `"UNK"` is the default.

## Background

The HCWD metric was developed in *Beyond the Labels: A Comparative Study of
Local and Global Approaches to Hierarchical Image Classification* (University of
Stirling, 2026), which compared flat, multi-head and sequential architectures on
a 42,000-image, 265-species taxonomy with variable depth and a 12.8 imbalance
ratio, all sharing a frozen MobileNetV2 backbone.

The finding that motivated this package: the flat baseline looked competitive on
per-level accuracy, on set-based F1, and on mean taxonomic distance — and was
the only model to score negative under HCWD at a strict penalty weight. Its
errors were confident and taxonomically distant, and nothing else in the
standard toolkit surfaced that.

## Limitations

- Tree taxonomies only. Directed acyclic graphs, where a node has several
  parents, are not supported.
- HCWD is unbounded, and its scale depends on taxonomy depth, so scores are not
  comparable across datasets — only across models on the same data.
- The metric assumes confidence is meaningful. A badly calibrated model will
  produce a misleading curve; pair it with a calibration check.
- `α` has no principled default. `0.5` is a reporting convention, not a derived
  value.
- Validated on one dataset. Treat it as a proof of concept, not a benchmark.

## Licence

MIT
