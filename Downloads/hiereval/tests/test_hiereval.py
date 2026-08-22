"""Tests for hiereval.

The ``test_matches_reference_*`` cases pin behaviour to the original
dissertation evaluation script, whose logic is reproduced here independently.
If they fail, the package and the published results have diverged.
"""

import pytest

from hiereval import (
    Taxonomy,
    classify_error,
    distance_stats,
    effective_depth,
    error_profile,
    evaluate,
    full_path_accuracy,
    hcwd_curve,
    hcwd_score,
    hierarchical_prf,
    inconsistency_rate,
    lca_depth,
    lca_index,
    mean_taxonomic_distance,
    per_level_scores,
    taxonomic_distance,
)

CAT = ["Animal", "Cat", "Siamese", "UNK"]
SIBLING = ["Animal", "Cat", "Bombay", "UNK"]
DOG = ["Animal", "Dog", "Papillon", "UNK"]
ROSE = ["Plant", "Flowering", "Rose", "UNK"]
SHALLOW = ["Animal", "Cat", "UNK", "UNK"]


# --- reference implementation (from the original evaluation script) --------

def _ref_depth(h):
    depth = 0
    for node in h:
        if node != "UNK":
            depth += 1
        else:
            break
    return depth


def _ref_lca(pred, true):
    lca = -1
    for level in range(min(len(pred), len(true))):
        if pred[level] == true[level] != "UNK":
            lca = level
        else:
            break
    return lca


def _ref_distance(pred, true):
    lca = _ref_lca(pred, true)
    return (_ref_depth(pred) - 1 - lca) + (_ref_depth(true) - 1 - lca)


def _ref_hcwd(pred, true, conf, alpha):
    delta = _ref_distance(pred, true)
    lca = _ref_lca(pred, true)
    return (lca * conf) / (1 + delta) - alpha * conf * delta


PAIRS = [(CAT, CAT), (CAT, SIBLING), (CAT, DOG), (CAT, ROSE), (SHALLOW, CAT)]


@pytest.mark.parametrize(("pred", "true"), PAIRS)
def test_matches_reference_distance(pred, true):
    assert taxonomic_distance(pred, true) == _ref_distance(pred, true)


@pytest.mark.parametrize(("pred", "true"), PAIRS)
@pytest.mark.parametrize("alpha", [0.1, 0.5, 1.0])
def test_matches_reference_hcwd(pred, true, alpha):
    got = hcwd_score([true], [pred], [0.87], alpha=alpha)
    assert got == pytest.approx(_ref_hcwd(pred, true, 0.87, alpha))


# --- paths -----------------------------------------------------------------

def test_effective_depth_stops_at_first_padding():
    assert effective_depth(CAT) == 3
    assert effective_depth(SHALLOW) == 2
    assert effective_depth(["UNK", "Cat"]) == 0


def test_lca_index_is_minus_one_when_nothing_shared():
    assert lca_index(CAT, ROSE) == -1
    assert lca_index(CAT, DOG) == 0
    assert lca_index(CAT, SIBLING) == 1
    assert lca_index(CAT, CAT) == 2


def test_lca_depth_is_a_count():
    assert lca_depth(CAT, ROSE) == 0
    assert lca_depth(CAT, CAT) == 3


def test_distance_is_zero_only_for_identical_paths():
    assert taxonomic_distance(CAT, CAT) == 0
    assert taxonomic_distance(CAT, SIBLING) == 2
    assert taxonomic_distance(CAT, DOG) == 4
    assert taxonomic_distance(CAT, ROSE) == 6


def test_distance_is_symmetric():
    assert taxonomic_distance(CAT, DOG) == taxonomic_distance(DOG, CAT)


def test_partial_depth_costs_one_edge():
    assert taxonomic_distance(SHALLOW, CAT) == 1


# --- HCWD ------------------------------------------------------------------

def test_correct_prediction_scores_higher_with_confidence():
    assert 0 < hcwd_score([CAT], [CAT], [0.2]) < hcwd_score([CAT], [CAT], [0.9])


def test_confident_error_is_punished_harder_than_hesitant_one():
    assert hcwd_score([CAT], [ROSE], [0.99]) < hcwd_score([CAT], [ROSE], [0.1])


def test_sibling_error_beats_cross_branch_at_equal_confidence():
    assert hcwd_score([CAT], [SIBLING], [0.8]) > hcwd_score([CAT], [ROSE], [0.8])


def test_score_falls_monotonically_with_alpha_when_errors_exist():
    curve = hcwd_curve([CAT, DOG], [SIBLING, ROSE], [0.9, 0.9])
    means = [v["mean"] for v in curve.values()]
    assert means == sorted(means, reverse=True)


def test_alpha_has_no_effect_on_a_perfect_model():
    curve = hcwd_curve([CAT], [CAT], [0.9])
    assert len({round(v["mean"], 12) for v in curve.values()}) == 1


def test_confidently_wrong_model_goes_negative():
    assert hcwd_score([CAT], [ROSE], [1.0], alpha=1.0) < 0


def test_per_level_confidence_uses_deepest_predicted_level():
    per_level = [[0.9, 0.8, 0.42, 0.1]]
    assert hcwd_score([CAT], [CAT], per_level) == pytest.approx(
        hcwd_score([CAT], [CAT], [0.42])
    )


def test_rejects_confidence_outside_unit_interval():
    with pytest.raises(ValueError, match="outside"):
        hcwd_score([CAT], [CAT], [1.4])


def test_rejects_mismatched_lengths():
    with pytest.raises(ValueError, match="same length"):
        hcwd_score([CAT, DOG], [CAT], [0.5, 0.5])


def test_rejects_negative_alpha():
    with pytest.raises(ValueError, match="non-negative"):
        hcwd_score([CAT], [CAT], [0.5], alpha=-1.0)


def test_curve_reports_standard_deviation():
    curve = hcwd_curve([CAT, CAT], [CAT, ROSE], [0.9, 0.9])
    assert curve[0.5]["std"] > 0


# --- set-based -------------------------------------------------------------

def test_hierarchical_prf_perfect():
    prf = hierarchical_prf([CAT, DOG], [CAT, DOG])
    assert prf.precision == prf.recall == prf.f1 == 1.0


def test_hierarchical_prf_gives_partial_credit():
    assert 0 < hierarchical_prf([CAT], [SIBLING]).f1 < 1


def test_samples_without_ground_truth_are_skipped():
    prf = hierarchical_prf([CAT, ["UNK"] * 4], [CAT, CAT])
    assert prf.n_skipped == 1
    assert prf.f1 == 1.0


def test_full_path_accuracy():
    assert full_path_accuracy([CAT, DOG], [CAT, ROSE]) == 0.5


def test_macro_distance_upweights_rare_classes():
    y_true, y_pred = [CAT, CAT, CAT, DOG], [CAT, CAT, CAT, ROSE]
    assert mean_taxonomic_distance(
        y_true, y_pred, average="macro"
    ) > mean_taxonomic_distance(y_true, y_pred, average="micro")


def test_rejects_unknown_average():
    with pytest.raises(ValueError, match="micro"):
        mean_taxonomic_distance([CAT], [CAT], average="weird")


# --- per-level -------------------------------------------------------------

def test_per_level_scores_shape_and_values():
    levels = per_level_scores([CAT, DOG], [CAT, DOG])
    assert [level.level for level in levels] == [1, 2, 3, 4]
    assert levels[0].accuracy == 1.0
    assert levels[3].n_valid == 0


def test_per_level_excludes_padded_ground_truth():
    levels = per_level_scores([SHALLOW, CAT], [SHALLOW, CAT])
    assert levels[2].n_valid == 1


def test_micro_equals_accuracy_for_single_label():
    level = per_level_scores([CAT, DOG], [CAT, ROSE])[0]
    assert level.precision_micro == level.recall_micro == level.accuracy


# --- errors ----------------------------------------------------------------

def test_error_categories():
    assert classify_error(CAT, CAT) == "exact"
    assert classify_error(CAT, SHALLOW) == "generalisation"
    assert classify_error(SHALLOW, CAT) == "specialisation"
    assert classify_error(CAT, SIBLING) == "sibling"
    assert classify_error(CAT, DOG) == "cross_branch"
    assert classify_error(CAT, ROSE) == "catastrophic"


def test_profiles_sum_to_one():
    y_true, y_pred = [CAT, DOG, CAT, ROSE], [CAT, ROSE, SIBLING, ROSE]
    assert sum(error_profile(y_true, y_pred).values()) == pytest.approx(1.0)


def test_distance_stats_distribution_counts_all_samples():
    stats = distance_stats([CAT, CAT], [CAT, ROSE])
    assert sum(stats.distribution.values()) == 2
    assert stats.perfect_matches == 1
    assert stats.perfect_rate == 0.5
    assert stats.maximum == 6


# --- taxonomy / TICE -------------------------------------------------------

def test_valid_predictions_have_zero_inconsistency():
    tax = Taxonomy.from_paths([CAT, DOG, ROSE])
    assert inconsistency_rate([CAT, DOG], tax) == 0.0


def test_impossible_path_is_flagged():
    tax = Taxonomy.from_paths([CAT, DOG, ROSE])
    assert inconsistency_rate([["Animal", "Dog", "Siamese", "UNK"]], tax) == 1.0


def test_inconsistency_rate_is_a_proportion():
    tax = Taxonomy.from_paths([CAT, DOG])
    preds = [CAT, ["Animal", "Cat", "Papillon", "UNK"]]
    assert inconsistency_rate(preds, tax) == 0.5


# --- report ----------------------------------------------------------------

def test_evaluate_runs_everything():
    y_true = [CAT, DOG, CAT, ROSE]
    y_pred = [CAT, ROSE, SIBLING, ROSE]
    conf = [0.9, 0.95, 0.6, 0.8]
    report = evaluate(
        y_true, y_pred, conf, name="test", taxonomy=Taxonomy.from_paths(y_true)
    )
    assert report.n_samples == 4
    assert len(report.per_level) == 4
    assert report.hcwd
    assert report.inconsistency_rate is not None
    assert "test" in report.summary()


def test_evaluate_without_confidence_skips_hcwd():
    report = evaluate([CAT], [CAT])
    assert report.hcwd == {}
    assert report.inconsistency_rate is None
    assert report.summary()
