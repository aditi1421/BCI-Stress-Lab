"""Load only the pinned, audited recordings and extract identical clean epochs."""

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import mne
import numpy as np
from scipy.io import loadmat
from scipy.signal import butter, sosfilt

from bci_stress.config import CleanConfig


@dataclass(frozen=True)
class EpochSet:
    X: np.ndarray
    y: np.ndarray
    trials: list[dict]
    channels: tuple[str, ...]
    sampling_hz: float


@dataclass(frozen=True)
class PreparedData:
    train: EpochSet
    test: EpochSet
    audit: dict


def _filter_sos(sampling_hz: float, config: CleanConfig) -> np.ndarray:
    return butter(
        config.filter_order,
        [config.low_hz, config.high_hz],
        btype="bandpass",
        fs=sampling_hz,
        output="sos",
    )


def causal_filter(values: np.ndarray, sampling_hz: float, config: CleanConfig) -> np.ndarray:
    """Filter a continuous run forward once, with zero state and no future padding."""
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError("Expected finite continuous data shaped (channels, samples)")
    return sosfilt(_filter_sos(sampling_hz, config), values, axis=-1)


def _fingerprint(path: Path) -> dict:
    with path.open("rb") as source:
        digest = hashlib.file_digest(source, "sha256").hexdigest()
    return {"size": path.stat().st_size, "sha256": digest}


def _verify_files(data_root: Path, entries: list[dict]) -> list[dict]:
    verified = []
    for entry in entries:
        path = data_root / entry["path"]
        if not path.resolve().is_relative_to(data_root.resolve()):
            raise ValueError("Manifest file must remain inside the dataset root")
        actual = _fingerprint(path)
        if actual != {"size": entry["size"], "sha256": entry["sha256"]}:
            raise ValueError(f"Dataset checksum or size mismatch: {entry['path']}")
        verified.append({"path": entry["path"], **actual})
    return verified


def _edf_signal_units(path: Path) -> list[str]:
    with path.open("rb") as source:
        source.seek(252)
        channel_count = int(source.read(4))
        source.seek(256)
        labels = [source.read(16).decode("ascii").strip() for _ in range(channel_count)]
        source.seek(256 + 96 * channel_count)
        units = [source.read(8).decode("ascii").strip() for _ in range(channel_count)]
    return [unit for label, unit in zip(labels, units, strict=True) if label != "EDF Annotations"]


def _sample(time_s: float, sampling_hz: float) -> int:
    value = time_s * sampling_hz
    rounded = round(value)
    if not np.isclose(value, rounded, rtol=0, atol=1e-7):
        raise ValueError("Event or epoch time does not align with a sample")
    return rounded


def _extract_epochs(
    filtered: np.ndarray,
    events: list[dict],
    channels: tuple[str, ...],
    sampling_hz: float,
    run: int,
    config: CleanConfig,
) -> EpochSet:
    epochs, labels, trials = [], [], []
    label_map = {"TASK2T1": 0, "TASK2T2": 1}
    start_offset = _sample(config.epoch_start_s, sampling_hz)
    stop_offset = _sample(config.epoch_stop_s, sampling_hz)
    for event in events:
        if event["label"] == "TASK2T0":
            continue
        if event["label"] not in label_map:
            raise ValueError(f"Unexpected event label: {event['label']}")
        event_sample = event["sample"]
        start_sample, stop_sample = event_sample + start_offset, event_sample + stop_offset
        annotated_stop = event_sample + _sample(event["duration_s"], sampling_hz)
        if stop_sample > annotated_stop or stop_sample > filtered.shape[-1]:
            raise ValueError("Epoch extends beyond annotated imagery or recording end")
        if start_sample < _sample(config.warmup_s, sampling_hz):
            raise ValueError("Epoch starts before the configured filter warmup ends")
        epochs.append(filtered[:, start_sample:stop_sample].copy())
        labels.append(label_map[event["label"]])
        trials.append({
            "id": f"sub-001_run-{run}_sample-{event_sample}",
            "run": run,
            "event_sample": event_sample,
            "start_sample": start_sample,
            "stop_sample": stop_sample,
            "onset_s": event_sample / sampling_hz,
            "label": event["label"],
            "target": label_map[event["label"]],
            "imagery_duration_s": event["duration_s"],
        })
    if not epochs:
        raise ValueError("Recording contains no left/right imagery epochs")
    values = np.stack(epochs)
    targets = np.asarray(labels, dtype=np.int64)
    values.setflags(write=False)
    targets.setflags(write=False)
    return EpochSet(values, targets, trials, channels, sampling_hz)


def _read_run(data_root: Path, manifest: dict, run: int, config: CleanConfig) -> tuple[EpochSet, dict]:
    set_path = data_root / f"sub-001/eeg/sub-001_task-motion_run-{run}_eeg.set"
    edf_path = data_root / f"sourcedata/rawdata/S001/S001R{run:02d}.edf"
    recording = loadmat(set_path, simplify_cells=True)
    recording = recording.get("EEG", recording)
    channels = tuple(channel["labels"] for channel in recording["chanlocs"])
    sampling_hz = float(recording["srate"])
    values_uv = np.asarray(recording["data"])
    expected_shape = (len(manifest["channels"]), manifest["samples_per_run"])
    if (
        channels != tuple(manifest["channels"])
        or sampling_hz != manifest["sampling_hz"]
        or values_uv.shape != expected_shape
        or not np.isfinite(values_uv).all()
        or recording["nbchan"] != len(channels)
        or recording["pnts"] != expected_shape[1]
        or recording["run"] != run
        or recording["session"] != 1
        or recording["trials"] != 1
    ):
        raise ValueError(f"Run {run} does not match the audited recording layout")
    expected_times_ms = np.arange(expected_shape[1]) * 1000.0 / sampling_hz
    if not np.array_equal(recording["times"], expected_times_ms):
        raise ValueError(f"Unexpected SET sample times in run {run}")
    if _edf_signal_units(edf_path) != ["uV"] * len(channels):
        raise ValueError(f"Source EDF does not explicitly use microvolts in run {run}")
    source = mne.io.read_raw_edf(edf_path, preload=True, verbose="ERROR")
    source_names = tuple(name.rstrip(".").casefold() for name in source.ch_names)
    if (
        source_names != tuple(name.casefold() for name in channels)
        or source.info["sfreq"] != sampling_hz
        or source.n_times != expected_shape[1]
        or set(source.get_channel_types()) != {"eeg"}
    ):
        raise ValueError(f"Source EDF layout disagrees with SET in run {run}")
    values = values_uv.astype(np.float64) * 1e-6
    if not np.array_equal(values, source.get_data()):
        raise ValueError(f"Source EDF and SET signals disagree after unit conversion in run {run}")
    set_events = recording["event"]
    if len(set_events) != 30 or len(source.annotations) != 30:
        raise ValueError(f"Expected all 30 audited events in run {run}")
    events = []
    for index, (event, annotation) in enumerate(zip(set_events, source.annotations, strict=True)):
        label = event["type"]
        latency = float(event["latency"]) - 1
        event_sample = round(latency)
        expected_sample = (index // 2) * 1328 + (672 if index % 2 else 0)
        expected_duration = 4.1 if index % 2 else 4.2
        if (
            latency != event_sample
            or event_sample != expected_sample
            or event_sample != _sample(float(annotation["onset"]), sampling_hz)
            or label != "TASK2" + annotation["description"]
            or (label == "TASK2T0") != (index % 2 == 0)
            or label not in {"TASK2T0", "TASK2T1", "TASK2T2"}
            or not np.isclose(annotation["duration"], expected_duration, rtol=0, atol=1e-8)
        ):
            raise ValueError(f"SET/source EDF event disagreement in run {run}, event {index}")
        events.append({
            "sample": event_sample,
            "onset_s": event_sample / sampling_hz,
            "label": label,
            "duration_s": float(annotation["duration"]),
        })
    filtered = causal_filter(values, sampling_hz, config)
    epochs = _extract_epochs(filtered, events, channels, sampling_hz, run, config)
    counts = {"left": int(np.sum(epochs.y == 0)), "right": int(np.sum(epochs.y == 1))}
    if counts != manifest["expected_counts"][str(run)]:
        raise ValueError(f"Run {run} imagery counts disagree with the audited manifest")
    audit = {
        "run": run,
        "session": 1,
        "sampling_hz": sampling_hz,
        "channels": list(channels),
        "samples": expected_shape[1],
        "duration_s": expected_shape[1] / sampling_hz,
        "stored_units": "uV",
        "working_units": "V",
        "edf_set_samples_equal_in_volts": True,
        "set_reference_field": str(recording["ref"]),
        "imagery_counts": counts,
        "rest_events": 15,
        "exclusions": [],
        "events": events,
        "unannotated_tail_s": 0.5,
    }
    return epochs, audit


def prepare_data(data_root: Path, manifest_path: Path, config: CleanConfig) -> PreparedData:
    """Verify all source bytes, process runs separately, and freeze the predefined split."""
    manifest = json.loads(manifest_path.read_text())
    expected_paths = {
        path
        for run in (*config.train_runs, config.test_run)
        for path in (
            f"sub-001/eeg/sub-001_task-motion_run-{run}_eeg.set",
            f"sourcedata/rawdata/S001/S001R{run:02d}.edf",
        )
    }
    if (
        manifest["dataset"] != "ds004362"
        or manifest["subject"] != "sub-001"
        or manifest["commit"] != config.dataset_commit
        or len(manifest["channels"]) != 64
        or len(set(manifest["channels"])) != 64
        or manifest["sampling_hz"] != 160
        or manifest["samples_per_run"] != 20000
        or len(manifest["files"]) != len(expected_paths)
        or {entry["path"] for entry in manifest["files"]} != expected_paths
    ):
        raise ValueError("Manifest does not describe the pinned sub-001 runs 4, 8 and 12")
    files_before = _verify_files(data_root, manifest["files"])
    prepared_runs, audits = {}, []
    for run in (*config.train_runs, config.test_run):
        prepared_runs[run], run_audit = _read_run(data_root, manifest, run, config)
        audits.append(run_audit)
    files_after = _verify_files(data_root, manifest["files"])
    if files_before != files_after:
        raise ValueError("Source recordings changed while preparing epochs")
    train_runs = [prepared_runs[run] for run in config.train_runs]
    train_values = np.concatenate([epochs.X for epochs in train_runs])
    train_targets = np.concatenate([epochs.y for epochs in train_runs])
    train_values.setflags(write=False)
    train_targets.setflags(write=False)
    train = EpochSet(
        train_values,
        train_targets,
        [trial for epochs in train_runs for trial in epochs.trials],
        train_runs[0].channels,
        train_runs[0].sampling_hz,
    )
    test = prepared_runs[config.test_run]
    if {trial["id"] for trial in train.trials} & {trial["id"] for trial in test.trials}:
        raise ValueError("Training and test trial identities overlap")
    audit = {
        "dataset": manifest["dataset"],
        "subject": manifest["subject"],
        "dataset_commit": manifest["commit"],
        "manifest_sha256": _fingerprint(manifest_path)["sha256"],
        "files": files_before,
        "source_files_unchanged": True,
        "runs": audits,
        "train_trial_count": len(train.trials),
        "test_trial_count": len(test.trials),
        "preprocessing": {
            "filter": "scipy.signal.butter + sosfilt",
            "prototype_order": config.filter_order,
            "realized_bandpass_order": config.filter_order * 2,
            "sos": _filter_sos(train.sampling_hz, config).tolist(),
            "direction": "forward causal",
            "initial_state": "zero, reset separately for each continuous run",
            "padding": "none",
            "reference": "recorded reference retained; ear unspecified",
            "resampling": "none",
            "baseline_correction": "none",
            "epoch_interval": "[cue+1s, cue+4s)",
            "epoch_samples": train.X.shape[-1],
            "warmup_s": config.warmup_s,
            "artifact_rejection": "none",
            "channel_selection": "all 64, audited original order",
        },
    }
    return PreparedData(train, test, audit)
