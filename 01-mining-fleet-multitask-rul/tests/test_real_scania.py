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


def _histogram_readouts(scale: float = 1.0):
    rng = np.random.default_rng(0)
    rows = []
    for v in (1, 2):
        t = np.cumsum(rng.uniform(1, 4, 12))
        c = np.cumsum(rng.uniform(0, 5, 12))
        bins = np.cumsum(rng.uniform(0, 3, (12, 3)), axis=0)
        for i in range(12):
            rows.append({"vehicle_id": v, "time_step": float(t[i]), "171_0": c[i] * scale,
                         "397_0": bins[i, 0] * scale, "397_1": bins[i, 1] * scale, "397_2": bins[i, 2] * scale})
    return pl.DataFrame(rows)


def test_scale_invariant_features_do_not_move_when_a_variable_is_rescaled():
    from src.real_scania import feature_sets

    specs = pl.DataFrame({"vehicle_id": [1, 2], "Spec_0": ["Cat0", "Cat1"]})
    base = features(_histogram_readouts(1.0), specs).sort(["vehicle_id", "time_step"])
    scaled = features(_histogram_readouts(6.5), specs).sort(["vehicle_id", "time_step"])
    sets = feature_sets(base.columns)
    inv = [c for c in sets["scale_invariant"] if c not in ("time_step", "n_readouts")]
    assert {"rel_171_0", "frac_397_0", "frac_397_2"} <= set(inv) and "frac_171_0" not in inv
    assert np.allclose(base.select(inv).to_numpy(), scaled.select(inv).to_numpy(), equal_nan=True)
    assert not np.allclose(base["rate_171_0"].to_numpy(), scaled["rate_171_0"].to_numpy())  # the rates do move


def test_scale_invariant_features_never_look_ahead():
    specs = pl.DataFrame({"vehicle_id": [1, 2], "Spec_0": ["Cat0", "Cat1"]})
    r = _histogram_readouts()
    full = features(r, specs).sort(["vehicle_id", "time_step"]).filter(pl.col("vehicle_id") == 1)
    cut = r.filter((pl.col("vehicle_id") != 1) | (pl.col("time_step") <= r.filter(pl.col("vehicle_id") == 1)["time_step"][6]))
    early = features(cut, specs).sort(["vehicle_id", "time_step"]).filter(pl.col("vehicle_id") == 1)
    cols = ["rel_171_0", "frac_397_1"]
    assert np.allclose(full.head(7).select(cols).to_numpy(), early.select(cols).to_numpy(), equal_nan=True)
