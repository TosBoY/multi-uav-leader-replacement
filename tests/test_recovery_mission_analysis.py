import pandas as pd

from analysis.metrics_analysis.mission_success_analysis import (
    SAFETY_DISTANCE_M,
    T_MAX_S,
    add_mission_success_columns,
    classify_mission_tier,
    rank_mission_runs,
    _condition_summary,
)


def base_row(**overrides):
    row = {
        "run_id": 1,
        "number_of_drone": 5,
        "min_separation": SAFETY_DISTANCE_M,
        "n_matched_01": 4,
        "n_never_matched_01": 0,
        "n_unstable_01": 0,
        "time_to_match_max_01": 3.0,
        "settle_time_01": 3.0,
        "n_matched_002": 4,
        "n_never_matched_002": 0,
        "n_unstable_002": 0,
        "time_to_match_max_002": 3.5,
        "settle_time_002": 3.5,
        "error_integral_sum": 10.0,
    }
    row.update(overrides)
    return row


def test_unsafe_takes_priority_over_full_recovery():
    assert classify_mission_tier(
        base_row(min_separation=SAFETY_DISTANCE_M - 0.01)
    ) == 0


def test_missing_separation_is_unsafe():
    assert classify_mission_tier(base_row(min_separation=None)) == 0


def test_no_recovery_when_no_drone_matches_by_deadline():
    assert classify_mission_tier(
        base_row(
            n_matched_01=0,
            n_never_matched_01=4,
            time_to_match_max_01=None,
        )
    ) == 1


def test_partial_recovery_for_mixed_drone_statuses():
    assert classify_mission_tier(
        base_row(n_matched_01=3, n_never_matched_01=1)
    ) == 2


def test_late_recovery_is_not_full():
    assert classify_mission_tier(
        base_row(time_to_match_max_01=T_MAX_S + 0.1)
    ) == 1


def test_full_coarse_and_full_fine_tiers():
    assert classify_mission_tier(base_row()) == 4
    assert classify_mission_tier(
        base_row(n_matched_002=3, n_never_matched_002=1)
    ) == 3


def test_ranking_is_tier_first_then_settle_then_integral():
    frame = pd.DataFrame([
        base_row(run_id=1, mission_tier=None, settle_time_01=3.0,
                  settle_time_002=3.5, error_integral_sum=20.0),
        base_row(run_id=2, settle_time_01=2.0,
                  settle_time_002=2.5, error_integral_sum=30.0),
        base_row(run_id=3, n_matched_01=3, n_never_matched_01=1,
                  settle_time_01=1.0, settle_time_002=None,
                  error_integral_sum=5.0),
        base_row(run_id=4, min_separation=0.1,
                  settle_time_01=0.5, settle_time_002=0.5,
                  error_integral_sum=1.0),
    ])
    ranked = rank_mission_runs(frame)
    assert ranked["mission_tier"].tolist() == [4, 4, 2, 0]
    assert ranked["run_id"].tolist() == [2, 1, 3, 4]

    classified = add_mission_success_columns(frame)
    assert classified["mission_tier_name"].tolist() == [
        "Full, fine",
        "Full, fine",
        "Partial recovery",
        "Unsafe",
    ]


def test_condition_summary_includes_supported_scenario_dimensions():
    frame = pd.DataFrame([
        base_row(
            run_id=1,
            shape="arrow",
            spread=1.0,
            number_of_drone=5,
            election_type="closest",
            difficulty_band="easy",
        ),
        base_row(
            run_id=2,
            shape="line",
            spread=2.0,
            number_of_drone=8,
            election_type="random",
            difficulty_band="hard",
            n_matched_01=3,
            n_never_matched_01=1,
        ),
    ])
    frame = add_mission_success_columns(frame)
    summary = _condition_summary(frame)

    conditions = set(summary["condition"])
    assert conditions == {
        "shape",
        "plane",
        "spread",
        "number_of_drone",
        "difficulty_band",
    }
