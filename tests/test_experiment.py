"""Checks for split isolation, paired permutation diagnostics, and saved artifacts."""

import json
from dataclasses import replace

import numpy as np
import pytest

from bci_stress import experiment
from bci_stress.artifacts import array_digest
from bci_stress.config import CleanConfig
from bci_stress.data import EpochSet, PreparedData


@pytest.fixture
def prepared():
    generator = np.random.default_rng(57)
    labels = np.tile([0, 1], 12)
    values = generator.normal(size=(24, 6, 80)) * 1e-6
    values[labels == 0, :2] *= 3
    values[labels == 1, 2:4] *= 3
    channels = tuple(f"channel-{index}" for index in range(6))
    train_trials = [
        {"id": f"train-{index}", "run": 4 if index < 8 else 8, "onset_s": index * 8.3}
        for index in range(16)
    ]
    test_trials = [
        {"id": f"test-{index}", "run": 12, "onset_s": index * 8.3}
        for index in range(8)
    ]
    return PreparedData(
        EpochSet(values[:16], labels[:16], train_trials, channels, 160),
        EpochSet(values[16:], labels[16:], test_trials, channels, 160),
        {"fixture": "synthetic"},
    )


def test_split_checks_reject_actual_data_and_identity_leaks(prepared):
    config = CleanConfig()
    assert experiment.check_split(prepared, config)["disjoint_runs"]
    overlap = prepared.test.trials.copy()
    overlap[0] = {**overlap[0], "id": prepared.train.trials[0]["id"]}
    with pytest.raises(ValueError, match="overlap"):
        experiment.check_split(replace(prepared, test=replace(prepared.test, trials=overlap)), config)
    copied_values = prepared.test.X.copy()
    copied_values[0] = prepared.train.X[0]
    with pytest.raises(ValueError, match="Identical signal"):
        experiment.check_split(replace(prepared, test=replace(prepared.test, X=copied_values)), config)
    with pytest.raises(ValueError, match="channel order"):
        experiment.check_split(
            replace(prepared, test=replace(prepared.test, channels=prepared.test.channels[::-1])),
            config,
        )


def test_permutations_fit_only_training_and_keep_counts_per_run(prepared, monkeypatch):
    calls = []

    class FitRecorder:
        def fit(self, values, labels):
            assert values is prepared.train.X
            calls.append(labels.copy())
            return self

        def predict(self, values):
            assert values is prepared.test.X
            return np.zeros(len(values), dtype=int)

    monkeypatch.setattr(experiment, "make_pipelines", lambda config: {
        "csp_lda": FitRecorder(), "bandpower_lda": FitRecorder(),
    })
    config = CleanConfig(permutation_count=5)
    observed = {name: {"balanced_accuracy": 0.75} for name in ("csp_lda", "bandpower_lda")}
    before = prepared.train.y.copy(), prepared.test.y.copy()
    first = experiment.permutation_diagnostic(prepared.train, prepared.test, config, observed, lambda _: None)
    second = experiment.permutation_diagnostic(prepared.train, prepared.test, config, observed, lambda _: None)
    assert first == second
    for index in range(0, len(calls), 2):
        np.testing.assert_array_equal(calls[index], calls[index + 1])
        for run_slice in (slice(0, 8), slice(8, 16)):
            np.testing.assert_array_equal(
                np.bincount(calls[index][run_slice]), np.bincount(prepared.train.y[run_slice]),
            )
    np.testing.assert_array_equal(prepared.train.y, before[0])
    np.testing.assert_array_equal(prepared.test.y, before[1])
    assert all(row["csp_lda"] == 0.5 for row in first["iterations"])


def test_observed_fit_has_no_test_argument_and_uses_same_arrays(prepared, monkeypatch):
    calls = []

    class FitRecorder:
        def fit(self, values, labels):
            calls.append((values, labels))
            return self

    monkeypatch.setattr(experiment, "make_pipelines", lambda config: {
        "csp_lda": FitRecorder(), "bandpower_lda": FitRecorder(),
    })
    experiment.fit_decoders(prepared.train, CleanConfig())
    assert len(calls) == 2
    assert all(values is prepared.train.X and labels is prepared.train.y for values, labels in calls)


def test_metric_axes_and_balanced_accuracy():
    result = experiment.score_predictions(np.array([0, 0, 0, 1]), np.array([0, 0, 1, 1]))
    assert result["confusion_matrix"] == [[2, 1], [0, 1]]
    assert result["accuracy"] == 0.75
    assert result["balanced_accuracy"] == pytest.approx(5 / 6)


def test_complete_run_saves_reproducible_artifacts(prepared, monkeypatch, tmp_path):
    monkeypatch.setattr(experiment, "prepare_data", lambda *args: prepared)
    manifest = tmp_path / "manifest.json"
    manifest.write_text('{"fixture": true}')
    config = CleanConfig(permutation_count=2)
    output = tmp_path / "result"
    metrics = experiment.run_experiment(config, tmp_path, manifest, output, tmp_path, lambda _: None)
    assert json.loads((output / "status.json").read_text())["status"] == "complete"
    assert json.loads((output / "config.json").read_text()) == config.model_dump(mode="json")
    assert json.loads((output / "metrics.json").read_text()) == metrics
    leakage = json.loads((output / "leakage_checks.json").read_text())
    assert leakage["inputs_unchanged_after_all_fits"]
    assert leakage["train_X_sha256"] == array_digest(prepared.train.X)
    assert (output / "confusion_matrices.json").is_file()
    assert len((output / "predictions.csv").read_text().splitlines()) == 17
    with pytest.raises(FileExistsError):
        experiment.run_experiment(config, tmp_path, manifest, output, tmp_path)


def test_failed_run_is_visibly_incomplete(monkeypatch, tmp_path):
    def broken_data(*args):
        raise ValueError("checksum failed")

    monkeypatch.setattr(experiment, "prepare_data", broken_data)
    manifest = tmp_path / "manifest.json"
    manifest.write_text("{}")
    output = tmp_path / "failed"
    with pytest.raises(ValueError, match="checksum failed"):
        experiment.run_experiment(CleanConfig(), tmp_path, manifest, output, tmp_path, lambda _: None)
    status = json.loads((output / "status.json").read_text())
    assert status["status"] == "failed"
    assert status["error_type"] == "ValueError"
    assert not (output / "metrics.json").exists()


@pytest.mark.parametrize("changes", [
    {"test_run": 8}, {"train_runs": [4, 12]}, {"epoch_stop_s": 4.1},
    {"low_hz": 31}, {"permutation_count": 0}, {"power_floor": float("nan")},
])
def test_config_rejects_changed_split_window_and_invalid_values(changes):
    with pytest.raises(ValueError):
        CleanConfig(**changes)
