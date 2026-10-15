from datetime import date

import pytest

from src.data.mining_data_generator import generate_mining_dataset
from src.features.engineering import engineer_features
from src.models.multi_task_net import build_multi_task_table, train_multi_task_model
from src.models.train_survival_pipeline import equipment_train_test_split


@pytest.fixture(scope="module")
def multi_task_table():
    equipment_metadata, sensor_telemetry, maintenance_logs = generate_mining_dataset(
        n_equipment=60, seed=11, reference_date=date(2026, 8, 20), target_readings=150
    )
    features = engineer_features(sensor_telemetry)
    return build_multi_task_table(features, equipment_metadata, maintenance_logs)


def test_multi_task_table_has_both_labels(multi_task_table):
    assert multi_task_table.height > 0
    assert "rul_hours" in multi_task_table.columns
    assert "failure_type" in multi_task_table.columns
    assert multi_task_table["failure_type"].null_count() == 0


def test_train_multi_task_model_produces_finite_metrics(multi_task_table):
    ids = multi_task_table["equipment_id"].unique().to_list()
    train_ids, test_ids = equipment_train_test_split(ids, test_size=0.3, seed=42)

    _, _, metrics = train_multi_task_model(multi_task_table, train_ids, test_ids, epochs=3, batch_size=32)

    assert metrics["mae_rul_hours"] >= 0
    assert metrics["mae_rul_hours"] == metrics["mae_rul_hours"]  # not NaN
    assert 0.0 <= metrics["accuracy_failure_type"] <= 1.0
    assert 0.0 <= metrics["f1_macro_failure_type"] <= 1.0
    assert metrics["n_train_rows"] > 0
    assert metrics["n_test_rows"] > 0


def test_validation_is_carved_from_train_equipment_and_resume_works(multi_task_table, tmp_path, monkeypatch):
    import pytest
    import torch

    ids = sorted(multi_task_table["equipment_id"].unique().to_list())
    train_ids, test_ids = set(ids[: int(len(ids) * 0.75)]), set(ids[int(len(ids) * 0.75):])
    kwargs = dict(epochs=5, min_epochs=5, patience=100, batch_size=32)
    _, _, ref = train_multi_task_model(multi_task_table, train_ids, test_ids, **kwargs)
    assert ref["n_val_equipment"] >= 1
    assert ref["n_val_rows"] > 0

    ckpt = tmp_path / "mt.ckpt"
    real_step = torch.optim.Adam.step
    calls = {"n": 0}

    def flaky_step(self, *args, **kw):
        calls["n"] += 1
        if calls["n"] > 3:
            raise KeyboardInterrupt
        return real_step(self, *args, **kw)

    monkeypatch.setattr(torch.optim.Adam, "step", flaky_step)
    with pytest.raises(KeyboardInterrupt):
        train_multi_task_model(multi_task_table, train_ids, test_ids, checkpoint_path=ckpt, **kwargs)
    monkeypatch.setattr(torch.optim.Adam, "step", real_step)

    _, _, resumed = train_multi_task_model(multi_task_table, train_ids, test_ids, checkpoint_path=ckpt, **kwargs)
    assert resumed["loss_history"]["val_total"] == pytest.approx(ref["loss_history"]["val_total"], rel=1e-5)
    assert resumed["mae_rul_hours"] == pytest.approx(ref["mae_rul_hours"], rel=1e-5)
