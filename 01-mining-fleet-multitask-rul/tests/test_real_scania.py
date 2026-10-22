"""Labels, features and the decision rule of the SCANIA comparison (no download)."""
import numpy as np
import polars as pl

from src.real_scania import COST_MATRIX, features, label, min_expected_cost, total_cost


def _readouts():
    return pl.DataFrame({"vehicle_id": [1, 1, 1, 2], "time_step": [10.0, 20.0, 60.0, 5.0],
                         "171_0": [100.0, 300.0, 900.0, 10.0]})


def test_labels_follow_the_official_windows_and_censoring():
    tte = pl.DataFrame({"vehicle_id": [1, 2], "duration": [65.0, 50.0], "event": [1, 0]})
    out = label(_readouts(), tte).sort(["vehicle_id", "time_step"])
    # truck 1 repaired at 65: 55, 45 and 5 time units left -> classes 0, 1, 4
    assert out["cls"].to_list()[:3] == [0, 1, 4]
    # truck 2 censored: no remaining time, class 0
    assert out["ttf"].to_list()[3] is None and out["cls"].to_list()[3] == 0


def test_features_use_rates_and_never_look_ahead():
    specs = pl.DataFrame({"vehicle_id": [1, 2], "Spec_0": ["Cat0", "Cat1"]})
    full = features(_readouts(), specs).sort(["vehicle_id", "time_step"])
    early = features(_readouts().filter(pl.col("time_step") <= 20), specs).sort(["vehicle_id", "time_step"])
    assert full["rate_171_0"].to_list()[:2] == early["rate_171_0"].to_list()[:2] == [10.0, 15.0]
    assert full["n_readouts"].to_list()[:3] == [0, 1, 2]


def test_min_expected_cost_alarms_on_cheap_false_alarms():
    # 10% chance of failing within 6 units: expected cost of "nothing" = 50 > 10 for "class 4"
    proba = np.array([[0.9, 0, 0, 0, 0.1], [1.0, 0, 0, 0, 0]])
    assert list(min_expected_cost(proba)) == [4, 0]
    assert total_cost(np.array([0, 4]), np.array([4, 0])) == COST_MATRIX[0, 4] + COST_MATRIX[4, 0]
