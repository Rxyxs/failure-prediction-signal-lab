from src.ingest import Snapshot, time_to_failure_minutes


def test_time_to_failure_zero_at_last_file():
    snap = Snapshot(experiment="1st_test", file_index=99, n_files_in_experiment=100, signal=None)
    assert time_to_failure_minutes(snap) == 0


def test_time_to_failure_decreases_with_file_index():
    early = Snapshot("1st_test", 0, 100, None)
    late = Snapshot("1st_test", 50, 100, None)
    assert time_to_failure_minutes(early) > time_to_failure_minutes(late)
