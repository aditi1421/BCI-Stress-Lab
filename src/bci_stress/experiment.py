"""Fixed later-run evaluation and training-label permutation diagnostics."""

import hashlib
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

import mne
import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix
from sklearn.pipeline import Pipeline
from threadpoolctl import threadpool_limits

from .artifacts import array_digest, source_metadata, write_csv, write_json
from .config import CleanConfig
from .data import EpochSet, PreparedData, prepare_data
from .pipeline import make_pipelines


def check_split(prepared: PreparedData, config: CleanConfig) -> dict:
    train, test = prepared.train, prepared.test
    train_ids = [trial["id"] for trial in train.trials]
    test_ids = [trial["id"] for trial in test.trials]
    if len(set(train_ids)) != len(train_ids) or len(set(test_ids)) != len(test_ids):
        raise ValueError("Duplicate trial identifiers")
    if set(train_ids) & set(test_ids):
        raise ValueError("Training and test trial IDs overlap")
    if {trial["run"] for trial in train.trials} != set(config.train_runs):
        raise ValueError("Training data do not match configured training runs")
    if {trial["run"] for trial in test.trials} != {config.test_run}:
        raise ValueError("Test data do not match configured test run")
    if train.channels != test.channels or train.sampling_hz != test.sampling_hz:
        raise ValueError("Train/test channel order or sampling differs")
    if train.X.shape[1:] != test.X.shape[1:]:
        raise ValueError("Train/test epoch dimensions differ")
    for split in (train, test):
        if len(split.X) != len(split.y) or len(split.trials) != len(split.y):
            raise ValueError("Epoch, label and trial counts differ")
        if set(np.unique(split.y)) != {0, 1}:
            raise ValueError("Both left/right classes must be present")
    train_hashes = {array_digest(epoch) for epoch in train.X}
    test_hashes = {array_digest(epoch) for epoch in test.X}
    if train_hashes & test_hashes:
        raise ValueError("Identical signal epoch appears in both splits")
    return {
        "disjoint_trial_ids": True,
        "disjoint_runs": True,
        "no_duplicate_epoch_across_splits": True,
        "same_channel_order_and_epoch_dimensions": True,
        "train_trials": train_ids,
        "test_trials": test_ids,
        "train_X_sha256": array_digest(train.X),
        "train_y_sha256": array_digest(train.y),
        "test_X_sha256": array_digest(test.X),
        "test_y_sha256": array_digest(test.y),
    }


def fit_decoders(train: EpochSet, config: CleanConfig) -> dict[str, Pipeline]:
    fitted = make_pipelines(config)
    for pipeline in fitted.values():
        pipeline.fit(train.X, train.y)
    return fitted


def fitted_arrays(pipeline: Pipeline) -> dict[str, np.ndarray]:
    classifier = pipeline.named_steps["lda"]
    arrays = {
        "lda_classes": classifier.classes_,
        "lda_coef": classifier.coef_,
        "lda_intercept": classifier.intercept_,
        "lda_means": classifier.means_,
        "lda_covariance": classifier.covariance_,
    }
    if "csp" in pipeline.named_steps:
        arrays.update({
            "csp_filters": pipeline.named_steps["csp"].filters_,
            "csp_patterns": pipeline.named_steps["csp"].patterns_,
        })
    return arrays


def fitted_digest(pipeline: Pipeline) -> str:
    return hashlib.sha256("".join(
        name + array_digest(values) for name, values in sorted(fitted_arrays(pipeline).items())
    ).encode()).hexdigest()


def score_predictions(labels: np.ndarray, predictions: np.ndarray) -> dict:
    matrix = confusion_matrix(labels, predictions, labels=[0, 1])
    return {
        "n_trials": len(labels),
        "accuracy": float(accuracy_score(labels, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "class_order": ["left", "right"],
        "confusion_matrix": matrix.tolist(),
        "confusion_matrix_axes": "rows=true, columns=predicted",
        "class_recall": {
            label: float(matrix[index, index] / matrix[index].sum())
            for index, label in enumerate(("left", "right"))
        },
    }


def evaluate_decoders(
    fitted: dict[str, Pipeline], test: EpochSet,
) -> tuple[dict, list[dict], dict]:
    metrics, predictions, checks = {}, [], {}
    for name, pipeline in fitted.items():
        before = fitted_digest(pipeline)
        predicted = pipeline.predict(test.X)
        probabilities = pipeline.predict_proba(test.X)
        individually = np.concatenate([
            pipeline.predict_proba(epoch[None, :, :]) for epoch in test.X
        ])
        if not np.allclose(probabilities, individually, rtol=1e-10, atol=1e-12):
            raise RuntimeError("Prediction depends on other test epochs")
        if before != fitted_digest(pipeline):
            raise RuntimeError("Prediction mutated learned parameters")
        metrics[name] = score_predictions(test.y, predicted)
        checks[name] = {"learned_state_unchanged": True, "prediction_batch_independent": True}
        for index, trial in enumerate(test.trials):
            predictions.append({
                "decoder": name, **trial,
                "true_label": int(test.y[index]), "predicted_label": int(predicted[index]),
                "probability_left": float(probabilities[index, 0]),
                "probability_right": float(probabilities[index, 1]),
            })
    return metrics, predictions, checks


def permutation_diagnostic(
    train: EpochSet, test: EpochSet, config: CleanConfig,
    observed: dict, progress: Callable[[str], None] = print,
) -> dict:
    generator = np.random.default_rng(config.permutation_seed)
    run_ids = np.array([trial["run"] for trial in train.trials])
    results = []
    for iteration in range(config.permutation_count):
        shuffled = train.y.copy()
        for run in config.train_runs:
            indices = np.flatnonzero(run_ids == run)
            shuffled[indices] = generator.permutation(train.y[indices])
        row = {"iteration": iteration, "training_labels": shuffled.tolist()}
        for name, pipeline in make_pipelines(config).items():
            pipeline.fit(train.X, shuffled)
            predicted = pipeline.predict(test.X)
            row[name] = float(balanced_accuracy_score(test.y, predicted))
        results.append(row)
        if (iteration + 1) % 25 == 0 or iteration + 1 == config.permutation_count:
            progress(f"Label permutations: {iteration + 1}/{config.permutation_count}")
    summaries = {}
    for name in observed:
        scores = np.array([row[name] for row in results])
        score = observed[name]["balanced_accuracy"]
        summaries[name] = {
            "observed_balanced_accuracy": score,
            "null_mean": float(scores.mean()),
            "null_std": float(scores.std(ddof=0)),
            "null_quantiles_025_50_975": np.quantile(scores, [0.025, 0.5, 0.975]).tolist(),
            "diagnostic_tail_fraction_plus_one": float((1 + (scores >= score).sum()) / (1 + len(scores))),
        }
    return {
        "method": "Shuffle training labels independently within each training run; refit full pipelines",
        "test_labels": "Original run-12 labels held fixed; never used for fitting or tuning",
        "interpretation": (
            "Diagnostic null reference, not a calibrated significance test: temporal exchangeability "
            "is not established. Near-chance results do not prove absence of leakage."
        ),
        "seed": config.permutation_seed, "count": config.permutation_count,
        "summaries": summaries, "iterations": results,
    }


def run_experiment(
    config: CleanConfig, data_root: Path, manifest_path: Path, output: Path,
    project_root: Path, progress: Callable[[str], None] = print,
) -> dict:
    output.mkdir(parents=True, exist_ok=False)
    started = datetime.now(UTC).isoformat()
    write_json(output / "status.json", {"status": "running", "started_utc": started})
    try:
        write_json(output / "config.json", config.model_dump(mode="json"))
        write_json(output / "protocol.json", {
            "label_mapping": {"TASK2T1": 0, "TASK2T2": 1},
            "split": "sub-001 runs 4+8 training; run 12 fixed later-run test",
            "hyperparameter_search": False,
            "automatic_rejection": False,
            "reference": "preserved acquisition reference",
            "filter": "causal Butterworth SOS; zero state separately at each run start",
            "epoch": "half-open [cue+1, cue+4), no baseline correction, 480 samples",
            "pipeline_parameters": {
                name: {step_name: step.get_params(deep=False) for step_name, step in pipe.steps}
                for name, pipe in make_pipelines(config).items()
            },
        })
        metadata = source_metadata(project_root, output)
        metadata["started_utc"] = started
        metadata["config_sha256"] = hashlib.sha256((output / "config.json").read_bytes()).hexdigest()
        write_json(output / "metadata.json", metadata)
        (output / "dataset_manifest.json").write_bytes(manifest_path.read_bytes())
        progress("Validating source files and extracting the fixed train/test epochs")
        with threadpool_limits(limits=1), mne.use_log_level("ERROR"):
            prepared = prepare_data(data_root, manifest_path, config)
            leakage = check_split(prepared, config)
            write_json(output / "data_audit.json", prepared.audit)
            write_json(output / "split.json", {
                "train": prepared.train.trials, "test": prepared.test.trials,
                "channels": list(prepared.train.channels),
            })
            progress("Fitting both fixed pipelines on runs 4 and 8")
            fitted = fit_decoders(prepared.train, config)
            for name, pipeline in fitted.items():
                np.savez_compressed(output / f"{name}_parameters.npz", **fitted_arrays(pipeline))
            metrics, predictions, prediction_checks = evaluate_decoders(fitted, prepared.test)
            write_json(output / "metrics.json", metrics)
            write_csv(output / "predictions.csv", predictions)
            write_json(output / "confusion_matrices.json", {
                "class_order": ["left", "right"], "axes": "rows=true, columns=predicted",
                "matrices": {name: scores["confusion_matrix"] for name, scores in metrics.items()},
            })
            permutations = permutation_diagnostic(
                prepared.train, prepared.test, config, metrics, progress,
            )
            write_json(output / "label_permutations.json", permutations)
            after = check_split(prepared, config)
            if leakage != after:
                raise RuntimeError("Fit or evaluation mutated the epoch data or labels")
            leakage["inputs_unchanged_after_all_fits"] = True
            leakage["fitting_scope"] = "All observed/permuted fits use training epochs only"
            leakage["prediction_checks"] = prediction_checks
            leakage["both_decoders_input_hashes"] = {
                name: {key: value for key, value in after.items() if key.endswith("sha256")}
                for name in fitted
            }
            write_json(output / "leakage_checks.json", leakage)
        finished = datetime.now(UTC).isoformat()
        write_json(output / "status.json", {
            "status": "complete", "started_utc": started, "finished_utc": finished,
        })
        progress(f"Complete: {output}")
        return metrics
    except Exception as error:
        write_json(output / "status.json", {
            "status": "failed", "started_utc": started,
            "error_type": type(error).__name__, "error": str(error),
        })
        raise
