"""Continuous test-record corruption with verified, frozen clean decoders."""

import csv
import hashlib
import json
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import mne
import numpy as np
from scipy.io import loadmat
from sklearn.pipeline import Pipeline
from threadpoolctl import threadpool_limits

from .artifacts import array_digest, source_metadata, write_csv, write_json
from .config import CleanConfig
from .corruption import add_gaussian_noise, flatline_channels, training_rms
from .data import (
    EpochSet,
    PreparedData,
    _extract_epochs,
    _verify_files,
    causal_filter,
    prepare_data,
)
from .experiment import check_split, evaluate_decoders, fit_decoders, fitted_arrays, fitted_digest
from .stress_config import StressConfig


@dataclass(frozen=True)
class StressInputs:
    prepared: PreparedData
    native_runs_uv: dict[int, np.ndarray]
    scale_uv: np.ndarray
    baseline_metrics: dict


def input_fingerprints(inputs: StressInputs) -> dict:
    return {
        "native_runs_uv": {
            str(run): array_digest(values) for run, values in inputs.native_runs_uv.items()
        },
        "train_X": array_digest(inputs.prepared.train.X),
        "train_y": array_digest(inputs.prepared.train.y),
        "test_X": array_digest(inputs.prepared.test.X),
        "test_y": array_digest(inputs.prepared.test.y),
        "scale_uv": array_digest(inputs.scale_uv),
    }


def prepare_stress(
    data_root: Path, manifest_path: Path, clean_config: CleanConfig, baseline_dir: Path,
) -> StressInputs:
    reference_config = CleanConfig.load(baseline_dir / "config.json")
    if clean_config != reference_config:
        raise ValueError("Clean configuration differs from the fixed original baseline")
    if json.loads((baseline_dir / "status.json").read_text())["status"] != "complete":
        raise ValueError("Reference clean baseline is incomplete")
    prepared = prepare_data(data_root, manifest_path, clean_config)
    check_split(prepared, clean_config)
    native_runs = {}
    for run in (*clean_config.train_runs, clean_config.test_run):
        path = data_root / f"sub-001/eeg/sub-001_task-motion_run-{run}_eeg.set"
        recording = loadmat(path, simplify_cells=True)
        recording = recording.get("EEG", recording)
        values = np.array(recording["data"], dtype=np.float64, copy=True)
        values.setflags(write=False)
        native_runs[run] = values
    _verify_files(data_root, prepared.audit["files"])
    scale = training_rms([native_runs[run] for run in clean_config.train_runs])
    scale.setflags(write=False)
    return StressInputs(
        prepared, native_runs, scale,
        json.loads((baseline_dir / "metrics.json").read_text()),
    )


def epochs_from_native(
    native_uv: np.ndarray, inputs: StressInputs, clean_config: CleanConfig,
) -> EpochSet:
    original = inputs.prepared.test
    run_audit = next(
        audit for audit in inputs.prepared.audit["runs"] if audit["run"] == clean_config.test_run
    )
    filtered_volts = causal_filter(native_uv * 1e-6, original.sampling_hz, clean_config)
    epochs = _extract_epochs(
        filtered_volts, run_audit["events"], original.channels,
        original.sampling_hz, clean_config.test_run, clean_config,
    )
    if epochs.trials != original.trials or not np.array_equal(epochs.y, original.y):
        raise RuntimeError("Corruption changed the test trial identities or labels")
    return epochs


def reconstruct_frozen(
    inputs: StressInputs, clean_config: CleanConfig, baseline_dir: Path,
) -> tuple[dict[str, Pipeline], dict, dict]:
    before = input_fingerprints(inputs)
    fitted = fit_decoders(inputs.prepared.train, clean_config)
    for name, pipeline in fitted.items():
        with np.load(baseline_dir / f"{name}_parameters.npz", allow_pickle=False) as reference:
            actual = fitted_arrays(pipeline)
            if set(actual) != set(reference.files):
                raise RuntimeError(f"Learned parameter fields differ for {name}")
            for key, values in actual.items():
                if not np.array_equal(values, reference[key]):
                    raise RuntimeError(f"Reconstructed {name}/{key} differs from saved clean model")
    original_epochs = epochs_from_native(
        inputs.native_runs_uv[clean_config.test_run], inputs, clean_config,
    )
    if not np.array_equal(original_epochs.X, inputs.prepared.test.X):
        raise RuntimeError("Native-unit zero path differs from original clean epochs")
    metrics, predictions, checks = evaluate_decoders(fitted, original_epochs)
    if metrics != inputs.baseline_metrics:
        raise RuntimeError("Clean baseline metric reproduction failed")
    with (baseline_dir / "predictions.csv").open(newline="") as handle:
        reference_predictions = {
            (row["decoder"], row["id"]): row for row in csv.DictReader(handle)
        }
    if len(reference_predictions) != len(predictions):
        raise RuntimeError("Clean baseline prediction count differs")
    for row in predictions:
        reference = reference_predictions[(row["decoder"], row["id"])]
        for key in ("true_label", "predicted_label"):
            if row[key] != int(reference[key]):
                raise RuntimeError("Clean baseline predicted labels differ")
        for key in ("probability_left", "probability_right"):
            if row[key] != float(reference[key]):
                raise RuntimeError("Clean baseline predicted probabilities differ")
    if input_fingerprints(inputs) != before:
        raise RuntimeError("Clean reconstruction changed training data or other inputs")
    return fitted, metrics, {
        "original_config_equal": True,
        "all_learned_arrays_exactly_equal": True,
        "zero_epochs_exactly_equal": True,
        "baseline_metrics_exactly_equal": True,
        "baseline_predictions_exactly_equal": True,
        "prediction_checks": checks,
        "frozen_model_fingerprints": {name: fitted_digest(pipe) for name, pipe in fitted.items()},
    }


def evaluate_condition(
    inputs: StressInputs, fitted: dict[str, Pipeline], clean_config: CleanConfig,
    kind: str, severity: float, seed: int,
) -> tuple[list[dict], list[dict], dict]:
    before_inputs = input_fingerprints(inputs)
    before_models = {name: fitted_digest(pipe) for name, pipe in fitted.items()}
    original_uv = inputs.native_runs_uv[clean_config.test_run]
    channels = inputs.prepared.test.channels
    failed_names = ()
    if kind == "gaussian":
        corrupted = add_gaussian_noise(original_uv, inputs.scale_uv, severity, seed)
    elif kind == "flatline":
        if int(severity) != severity:
            raise ValueError("Flatline severity must be an integer channel count")
        corrupted, failed_names = flatline_channels(
            original_uv, channels, count=int(severity), seed=seed,
        )
    else:
        raise ValueError(f"Unknown corruption family: {kind}")
    epochs = epochs_from_native(corrupted, inputs, clean_config)
    zero_epoch_equal = None
    if severity == 0:
        zero_epoch_equal = np.array_equal(epochs.X, inputs.prepared.test.X)
        if not np.array_equal(corrupted, original_uv) or not zero_epoch_equal:
            raise RuntimeError("Zero corruption does not reproduce original data and epochs")
    metrics, predictions, prediction_checks = evaluate_decoders(fitted, epochs)
    if severity == 0 and metrics != inputs.baseline_metrics:
        raise RuntimeError("Zero corruption does not reproduce original metrics")
    if input_fingerprints(inputs) != before_inputs:
        raise RuntimeError("Stress evaluation mutated training data, test data, labels or scale")
    after_models = {name: fitted_digest(pipe) for name, pipe in fitted.items()}
    if after_models != before_models:
        raise RuntimeError("Stress evaluation changed learned model parameters")
    identifier = f"{kind}-{severity:g}-seed-{seed}"
    base = {
        "condition_id": identifier, "corruption": kind, "severity": severity,
        "seed": seed, "failed_channels": json.dumps(list(failed_names)),
    }
    metric_rows = []
    for decoder, scores in metrics.items():
        matrix = scores["confusion_matrix"]
        decoder_predictions = [row for row in predictions if row["decoder"] == decoder]
        predicted_classes = {row["predicted_label"] for row in decoder_predictions}
        metric_rows.append({
            **base, "decoder": decoder, "n_trials": scores["n_trials"],
            "accuracy": scores["accuracy"], "balanced_accuracy": scores["balanced_accuracy"],
            "delta_balanced_accuracy": (
                scores["balanced_accuracy"] - inputs.baseline_metrics[decoder]["balanced_accuracy"]
            ),
            "left_recall": scores["class_recall"]["left"],
            "right_recall": scores["class_recall"]["right"],
            "true_left_pred_left": matrix[0][0], "true_left_pred_right": matrix[0][1],
            "true_right_pred_left": matrix[1][0], "true_right_pred_right": matrix[1][1],
            "constant_prediction": len(predicted_classes) == 1,
            "constant_class": next(iter(predicted_classes)) if len(predicted_classes) == 1 else "",
            "epoch_sha256": array_digest(epochs.X),
        })
    prediction_rows = [{**base, **row} for row in predictions]
    condition = {
        **base, "failed_channels": list(failed_names),
        "failed_indices": [channels.index(name) for name in failed_names],
        "native_units": "uV", "continuous_shape": list(corrupted.shape),
        "corrupted_native_sha256": array_digest(corrupted),
        "epoch_sha256": array_digest(epochs.X),
        "zero_epochs_equal": zero_epoch_equal,
        "inputs_unchanged": True, "prediction_checks": prediction_checks,
        "model_before_sha256": before_models, "model_after_sha256": after_models,
        "confusion_matrices": {
            name: scores["confusion_matrix"] for name, scores in metrics.items()
        },
    }
    if kind == "gaussian":
        condition["requested_noise_std_uv"] = (severity * inputs.scale_uv).tolist()
        condition["realized_injected_rms_uv"] = np.sqrt(
            np.mean(np.square(corrupted - original_uv), axis=1),
        ).tolist()
    return metric_rows, prediction_rows, condition


def summarize_metrics(rows: list[dict]) -> list[dict]:
    groups = sorted({(row["corruption"], row["severity"], row["decoder"]) for row in rows})
    summaries = []
    for corruption, severity, decoder in groups:
        selected = [row for row in rows if (
            row["corruption"], row["severity"], row["decoder"]
        ) == (corruption, severity, decoder)]
        if len({row["seed"] for row in selected}) != len(selected):
            raise ValueError("Duplicate seeds in a severity/decoder summary")
        summary = {
            "corruption": corruption, "severity": severity, "decoder": decoder,
            "n_replicates": len(selected),
            "constant_prediction_replicates": sum(row["constant_prediction"] for row in selected),
        }
        for metric in ("accuracy", "balanced_accuracy", "delta_balanced_accuracy"):
            values = np.asarray([row[metric] for row in selected])
            summary.update({
                f"{metric}_mean": float(values.mean()),
                f"{metric}_std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
                f"{metric}_min": float(values.min()), f"{metric}_max": float(values.max()),
            })
        summaries.append(summary)
    return summaries


def run_stress(
    stress_config: StressConfig, clean_config: CleanConfig, data_root: Path,
    manifest_path: Path, baseline_dir: Path, output: Path, project_root: Path,
    progress: Callable[[str], None] = print,
) -> dict:
    output.mkdir(parents=True, exist_ok=False)
    started = datetime.now(UTC).isoformat()
    write_json(output / "status.json", {"status": "running", "started_utc": started})
    try:
        write_json(output / "stress_config.json", stress_config.model_dump(mode="json"))
        write_json(output / "clean_config.json", clean_config.model_dump(mode="json"))
        metadata = source_metadata(project_root, output)
        metadata["started_utc"] = started
        metadata["definition"] = "Native-uV continuous test corruption before unchanged clean preprocessing"
        metadata["document_sha256"] = {}
        for name in ("stress-protocol.md", "stress-review.md"):
            document = project_root / "reports" / name
            content = document.read_bytes()
            (output / name).write_bytes(content)
            metadata["document_sha256"][name] = hashlib.sha256(content).hexdigest()
        metadata["stress_config_sha256"] = hashlib.sha256((output / "stress_config.json").read_bytes()).hexdigest()
        metadata["clean_baseline_reference"] = "reports/clean-sub001"
        metadata["clean_reference_sha256"] = {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(baseline_dir.iterdir())
            if path.suffix in {".json", ".csv", ".npz"}
        }
        write_json(output / "metadata.json", metadata)
        (output / "dataset_manifest.json").write_bytes(manifest_path.read_bytes())
        with threadpool_limits(limits=1), mne.use_log_level("ERROR"):
            inputs = prepare_stress(data_root, manifest_path, clean_config, baseline_dir)
            before = input_fingerprints(inputs)
            progress("Reconstructing and verifying the original frozen clean models")
            fitted, baseline_metrics, baseline_checks = reconstruct_frozen(
                inputs, clean_config, baseline_dir,
            )
            write_json(output / "clean_baseline_metrics.json", baseline_metrics)
            write_json(output / "baseline_checks.json", baseline_checks)
            write_json(output / "data_audit.json", inputs.prepared.audit)
            write_json(output / "split.json", {
                "train": inputs.prepared.train.trials, "test": inputs.prepared.test.trials,
                "channels": list(inputs.prepared.test.channels),
            })
            write_json(output / "training_scale.json", {
                "definition": stress_config.noise_scale,
                "units": "uV", "training_runs": list(clean_config.train_runs),
                "samples_per_channel": sum(len(values[0]) for run, values in inputs.native_runs_uv.items()
                                           if run in clean_config.train_runs),
                "channels": list(inputs.prepared.train.channels),
                "rms_uv": inputs.scale_uv.tolist(),
                "demeaned": False, "includes_rest": True, "filtered": False,
            })
            metric_rows, prediction_rows, conditions = [], [], []
            for kind, grid in (("gaussian", stress_config.noise_levels), ("flatline", stress_config.dropout_counts)):
                for severity in grid:
                    for seed in stress_config.seeds:
                        scores, predictions, condition = evaluate_condition(
                            inputs, fitted, clean_config, kind, severity, seed,
                        )
                        metric_rows.extend(scores)
                        prediction_rows.extend(predictions)
                        conditions.append(condition)
                    progress(f"Completed {kind}, severity {severity:g}, {len(stress_config.seeds)} seeds")
            if before != input_fingerprints(inputs):
                raise RuntimeError("Inputs changed during the stress experiment")
            _verify_files(data_root, inputs.prepared.audit["files"])
            summaries = summarize_metrics(metric_rows)
            write_csv(output / "metrics.csv", metric_rows)
            write_json(output / "metrics.json", metric_rows)
            write_csv(output / "predictions.csv", prediction_rows)
            write_json(output / "conditions.json", conditions)
            write_csv(output / "summary.csv", summaries)
            write_json(output / "summary.json", summaries)
            write_json(output / "validation.json", {
                "split": check_split(inputs.prepared, clean_config),
                "input_fingerprints_before": before,
                "input_fingerprints_after": input_fingerprints(inputs),
                "source_files_unchanged": True,
                "condition_count": len(conditions), "decoder_row_count": len(metric_rows),
                "prediction_row_count": len(prediction_rows),
                "zero_condition_count": sum(row["severity"] == 0 for row in conditions),
                "all_model_states_unchanged": all(
                    row["model_before_sha256"] == row["model_after_sha256"] for row in conditions
                ),
                "interpretation": "Seed spread is descriptive, conditional on one subject and 15 test trials",
            })
        write_json(output / "status.json", {
            "status": "complete", "started_utc": started,
            "finished_utc": datetime.now(UTC).isoformat(),
        })
        progress(f"Complete: {output}")
        return {"metrics": metric_rows, "summary": summaries}
    except (Exception, KeyboardInterrupt) as error:
        write_json(output / "status.json", {
            "status": "failed", "started_utc": started,
            "error_type": type(error).__name__, "error": str(error),
        })
        raise
