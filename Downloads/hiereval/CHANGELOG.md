# Changelog

All notable changes to this project are documented here.
This project adheres to [Semantic Versioning](https://semver.org/).

## [0.1.0] - unreleased

Initial release.

### Added
- `evaluate` / `Report` — run the whole framework in one call.
- `hcwd_score`, `hcwd_per_sample`, `hcwd_curve` — Confidence-Weighted
  Hierarchical Distance with penalty-weight sweeps.
- `taxonomic_distance`, `lca_index`, `lca_depth`, `effective_depth` — LCA-based
  path distance with variable-depth handling.
- `per_level_scores` — flat accuracy, precision, recall and F1 per level.
- `hierarchical_prf`, `full_path_accuracy`, `mean_taxonomic_distance`,
  `distance_stats`, `depth_breakdown` — set- and distance-based measures.
- `classify_error`, `error_profile`, `distance_distribution` — structured
  characterisation of failure modes.
- `Taxonomy`, `inconsistency_rate` — Tree-based Inconsistency Error (TICE).
