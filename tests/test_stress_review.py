"""Independent adversarial checks for the frozen continuous-corruption experiment."""

import json
from dataclasses import replace
from pathlib import Path

import mne
import numpy as np
import pytest
from threadpoolctl import threadpool_limits

from bci_stress import stress
from bci_stress.config import CleanConfig
from bci_stress.corruption import add_gaussian_noise, flatline_channels, training_rms
from bci_stress.data import EpochSet, PreparedData, _extract_epochs, causal_filter
from bci_stress.experiment import evaluate_decoders, fit_decoders, score_predictions
from bci_stress.pipeline import LogBandpower


def test_whole_run_flatline_reaches_existing_power_floor_without_nan():
    config = CleanConfig()
    continuous = np.random.default_rng(718).normal(size=(3, 2000))
    continuous[1] = 0
    filtered = causal_filter(continuous * 1e-6, 160, config)
    features = LogBandpower(config.power_floor).transform(filtered[None, :, 832:1312])
    np.testing.assert_array_equal(filtered[1], np.zeros(2000))
    assert features[0, 1] == np.log(config.power_floor)
    assert np.isfinite(features).all()


@pytest.mark.parametrize("constant_label", [0, 1])
def test_constant_prediction_is_chance_balanced_accuracy_despite_class_imbalance(constant_label):
    labels = np.array([0] * 7 + [1] * 8)
    metrics = score_predictions(labels, np.full_like(labels, constant_label))
    assert metrics["balanced_accuracy"] == 0.5
    expected_count = 7 if constant_label == 0 else 8
    assert metrics["accuracy"] == expected_count / 15
    assert sorted(metrics["class_recall"].values()) == [0.0, 1.0]


def test_metric_resolution_is_coarse_on_fifteen_heldout_trials():
    labels = np.array([0] * 7 + [1] * 8)
    left_error = labels.copy()
    left_error[0] = 1
    right_error = labels.copy()
    right_error[-1] = 0
    assert 1 - score_predictions(labels, left_error)["balanced_accuracy"] == pytest.approx(1 / 14)
    assert 1 - score_predictions(labels, right_error)["balanced_accuracy"] == pytest.approx(1 / 16)


@pytest.fixture
def synthetic_stress():
    config = CleanConfig()
    generator = np.random.default_rng(972)
    channels = ("C3", "C4", "Cz", "P3", "P4", "Fz")
    events = [
        {"sample": 160, "label": "TASK2T1", "duration_s": 4.1},
        {"sample": 1000, "label": "TASK2T2", "duration_s": 4.1},
        {"sample": 2000, "label": "TASK2T1", "duration_s": 4.1},
    ]
    native = {run: generator.normal(size=(6, 3200)) * 20 for run in (4, 8, 12)}
    epoch_runs = {
        run: _extract_epochs(causal_filter(values * 1e-6, 160, config), events,
                             channels, 160, run, config)
        for run, values in native.items()
    }
    train = EpochSet(
        np.concatenate([epoch_runs[run].X for run in (4, 8)]),
        np.concatenate([epoch_runs[run].y for run in (4, 8)]),
        [trial for run in (4, 8) for trial in epoch_runs[run].trials],
        channels, 160,
    )
    train.X.setflags(write=False)
    train.y.setflags(write=False)
    prepared = PreparedData(train, epoch_runs[12], {"runs": [{"run": 12, "events": events}]})
    scale = training_rms([native[4], native[8]])
    for values in native.values():
        values.setflags(write=False)
    with threadpool_limits(limits=1), mne.use_log_level("ERROR"):
        fitted = fit_decoders(train, config)
        baseline, _, _ = evaluate_decoders(fitted, prepared.test)
    return stress.StressInputs(prepared, native, scale, baseline), fitted, config


@pytest.mark.parametrize("kind,severity", [("gaussian", 0.5), ("flatline", 2)])
def test_synthetic_conditions_are_deterministic_frozen_and_input_preserving(
    synthetic_stress, monkeypatch, kind, severity,
):
    inputs, fitted, config = synthetic_stress
    before = stress.input_fingerprints(inputs)

    def forbidden_fit(*args, **kwargs):
        pytest.fail("Stress prediction attempted to fit a decoder")

    for pipeline in fitted.values():
        monkeypatch.setattr(pipeline, "fit", forbidden_fit)
        for step in pipeline.named_steps.values():
            monkeypatch.setattr(step, "fit", forbidden_fit)
    first = stress.evaluate_condition(inputs, fitted, config, kind, severity, 8)
    repeat = stress.evaluate_condition(inputs, fitted, config, kind, severity, 8)
    assert first == repeat
    assert stress.input_fingerprints(inputs) == before
    metrics, predictions, condition = first
    assert condition["model_before_sha256"] == condition["model_after_sha256"]
    assert condition["inputs_unchanged"]
    assert len({row["epoch_sha256"] for row in metrics}) == 1
    for row in metrics:
        assert row["delta_balanced_accuracy"] == (
            row["balanced_accuracy"] - inputs.baseline_metrics[row["decoder"]]["balanced_accuracy"]
        )
        selected = [item for item in predictions if item["decoder"] == row["decoder"]]
        independently_scored = score_predictions(
            np.array([item["true_label"] for item in selected]),
            np.array([item["predicted_label"] for item in selected]),
        )
        assert row["balanced_accuracy"] == independently_scored["balanced_accuracy"]
        assert row["accuracy"] == independently_scored["accuracy"]
        assert condition["confusion_matrices"][row["decoder"]] == (
            independently_scored["confusion_matrix"]
        )


@pytest.mark.parametrize("kind,severity", [("gaussian", 0.5), ("flatline", 2)])
def test_continuous_corruption_precedes_volt_filter_and_shared_epoch_predictions(
    synthetic_stress, monkeypatch, kind, severity,
):
    inputs, fitted, config = synthetic_stress
    seed = 13
    original = inputs.native_runs_uv[12]
    if kind == "gaussian":
        expected_native = add_gaussian_noise(original, inputs.scale_uv, severity, seed)
    else:
        expected_native, _ = flatline_channels(
            original, inputs.prepared.test.channels, count=severity, seed=seed,
        )
    filter_inputs = []
    prediction_inputs = []
    real_filter = stress.causal_filter

    def recording_filter(values, sampling_hz, settings):
        filter_inputs.append(values.copy())
        return real_filter(values, sampling_hz, settings)

    monkeypatch.setattr(stress, "causal_filter", recording_filter)
    for pipeline in fitted.values():
        actual_predict = pipeline.predict

        def recording_predict(values, actual_predict=actual_predict):
            prediction_inputs.append(values)
            return actual_predict(values)

        monkeypatch.setattr(pipeline, "predict", recording_predict)
    stress.evaluate_condition(inputs, fitted, config, kind, severity, seed)
    assert len(filter_inputs) == 1
    np.testing.assert_array_equal(filter_inputs[0], expected_native * 1e-6)
    assert filter_inputs[0].shape == original.shape
    assert len(prediction_inputs) == 2
    assert prediction_inputs[0] is prediction_inputs[1]
    expected_epochs = _extract_epochs(
        real_filter(expected_native * 1e-6, 160, config),
        inputs.prepared.audit["runs"][0]["events"],
        inputs.prepared.test.channels, 160, 12, config,
    )
    np.testing.assert_array_equal(prediction_inputs[0], expected_epochs.X)


def test_prepare_scale_reads_training_runs_only(synthetic_stress, monkeypatch, tmp_path):
    inputs, _, config = synthetic_stress
    baseline = tmp_path / "baseline"
    baseline.mkdir()
    (baseline / "config.json").write_text(config.model_dump_json())
    (baseline / "status.json").write_text('{"status": "complete"}')
    (baseline / "metrics.json").write_text(json.dumps(inputs.baseline_metrics))
    prepared = replace(inputs.prepared, audit={**inputs.prepared.audit, "files": []})
    native = {run: values.copy() for run, values in inputs.native_runs_uv.items()}
    monkeypatch.setattr(stress, "prepare_data", lambda *args: prepared)
    monkeypatch.setattr(stress, "_verify_files", lambda *args: [])

    def recording_loader(path, **kwargs):
        run = int(path.name.split("_run-")[1].split("_")[0])
        return {"data": native[run]}

    monkeypatch.setattr(stress, "loadmat", recording_loader)
    first = stress.prepare_stress(tmp_path, tmp_path / "manifest.json", config, baseline)
    native[12] *= 1e8
    altered = stress.prepare_stress(tmp_path, tmp_path / "manifest.json", config, baseline)
    expected = np.sqrt((np.sum(native[4] ** 2, axis=1) + np.sum(native[8] ** 2, axis=1)) / 6400)
    np.testing.assert_array_equal(first.scale_uv, altered.scale_uv)
    np.testing.assert_array_equal(first.scale_uv, expected)
    assert not first.scale_uv.flags.writeable


def test_changed_clean_configuration_is_rejected_before_source_loading(tmp_path, monkeypatch):
    (tmp_path / "config.json").write_text(CleanConfig().model_dump_json())

    def forbidden_load(*args):
        pytest.fail("Changed settings reached source loading")

    monkeypatch.setattr(stress, "prepare_data", forbidden_load)
    with pytest.raises(ValueError, match="configuration differs"):
        stress.prepare_stress(
            tmp_path, tmp_path / "manifest.json", CleanConfig(csp_components=3), tmp_path,
        )


def test_summary_keeps_absolute_scores_deltas_and_conditional_spread_separate():
    rows = [
        {
            "corruption": "flatline", "severity": 2, "decoder": "bandpower_lda",
            "seed": seed, "accuracy": score, "balanced_accuracy": score,
            "delta_balanced_accuracy": score - 0.4, "constant_prediction": seed == 0,
        }
        for seed, score in enumerate((0.5, 0.6))
    ]
    summary = stress.summarize_metrics(rows)[0]
    assert summary["n_replicates"] == 2
    assert summary["constant_prediction_replicates"] == 1
    assert summary["balanced_accuracy_mean"] == pytest.approx(0.55)
    assert summary["delta_balanced_accuracy_mean"] == pytest.approx(0.15)
    assert summary["balanced_accuracy_std"] == pytest.approx(np.std([0.5, 0.6], ddof=1))
    assert summary["balanced_accuracy_min"] == 0.5
    assert summary["balanced_accuracy_max"] == 0.6
    with pytest.raises(ValueError, match="Duplicate seeds"):
        stress.summarize_metrics([*rows, rows[0]])


@pytest.mark.integration
def test_real_zero_conditions_reproduce_all_saved_clean_outputs(monkeypatch):
    root = Path(__file__).resolve().parents[1]
    data_root = root / "data/ds004362"
    if not (data_root / "sub-001/eeg/sub-001_task-motion_run-4_eeg.set").is_file():
        pytest.skip("Audited dataset is not cached; this test never downloads data")
    config = CleanConfig.load(root / "configs/clean.json")
    baseline = root / "reports/clean-sub001"
    with threadpool_limits(limits=1), mne.use_log_level("ERROR"):
        inputs = stress.prepare_stress(
            data_root, root / "configs/sub-001-manifest.json", config, baseline,
        )
        fitted, metrics, checks = stress.reconstruct_frozen(inputs, config, baseline)
        assert metrics == json.loads((baseline / "metrics.json").read_text())
        assert checks["all_learned_arrays_exactly_equal"]
        assert checks["baseline_predictions_exactly_equal"]
        fingerprints = stress.input_fingerprints(inputs)
        expected_scale = np.sqrt(
            (np.sum(inputs.native_runs_uv[4] ** 2, axis=1)
             + np.sum(inputs.native_runs_uv[8] ** 2, axis=1)) / 40000
        )
        np.testing.assert_array_equal(inputs.scale_uv, expected_scale)

        def forbidden_fit(*args, **kwargs):
            pytest.fail("Zero stress condition attempted to fit a decoder")

        for pipeline in fitted.values():
            monkeypatch.setattr(pipeline, "fit", forbidden_fit)
            for step in pipeline.named_steps.values():
                monkeypatch.setattr(step, "fit", forbidden_fit)
        expected_predictions = evaluate_decoders(fitted, inputs.prepared.test)[1]
        for kind in ("gaussian", "flatline"):
            for seed in (0, 9):
                rows, predictions, condition = stress.evaluate_condition(
                    inputs, fitted, config, kind, 0, seed,
                )
                assert condition["zero_epochs_equal"]
                assert condition["failed_channels"] == []
                assert condition["model_before_sha256"] == condition["model_after_sha256"]
                assert all(row["delta_balanced_accuracy"] == 0 for row in rows)
                for actual, expected in zip(predictions, expected_predictions, strict=True):
                    assert all(actual[key] == value for key, value in expected.items())
                assert stress.input_fingerprints(inputs) == fingerprints
        assert stress._verify_files(data_root, inputs.prepared.audit["files"]) == (
            inputs.prepared.audit["files"]
        )
