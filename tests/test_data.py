import hashlib
from pathlib import Path

import numpy as np
import pytest
from scipy.signal import butter, sosfilt

from bci_stress.config import CleanConfig
from bci_stress.data import _extract_epochs, _sample, _verify_files, causal_filter, prepare_data


def test_filter_is_causal_and_does_not_mutate_input():
    config = CleanConfig()
    values = np.random.default_rng(71).normal(size=(3, 2000))
    original = values.copy()
    altered = values.copy()
    altered[:, 1000:] = 1e9
    filtered = causal_filter(values, 160, config)
    filtered_altered = causal_filter(altered, 160, config)
    np.testing.assert_array_equal(filtered[:, :1000], filtered_altered[:, :1000])
    np.testing.assert_array_equal(values, original)
    np.testing.assert_array_equal(filtered, causal_filter(values, 160, config))


def test_filter_matches_independent_zero_state_continuous_filter():
    config = CleanConfig()
    values = np.zeros((2, 2000))
    values[0, 100] = 1
    values[1, 1000] = 1
    expected = sosfilt(butter(4, [8, 30], btype="bandpass", fs=160, output="sos"), values)
    np.testing.assert_array_equal(causal_filter(values, 160, config), expected)
    np.testing.assert_array_equal(causal_filter(np.zeros_like(values), 160, config), 0)


def test_half_open_epochs_have_exact_samples_labels_and_source_ids():
    values = np.arange(3 * 2000, dtype=float).reshape(3, 2000)
    original = values.copy()
    events = [
        {"sample": 0, "label": "TASK2T0", "duration_s": 1.0},
        {"sample": 160, "label": "TASK2T1", "duration_s": 4.1},
        {"sample": 1000, "label": "TASK2T2", "duration_s": 4.1},
    ]
    epochs = _extract_epochs(values, events, ("C3", "Cz", "C4"), 160, 4, CleanConfig())
    assert epochs.X.shape == (2, 3, 480)
    np.testing.assert_array_equal(epochs.X[0], values[:, 320:800])
    np.testing.assert_array_equal(epochs.X[1], values[:, 1160:1640])
    np.testing.assert_array_equal(epochs.y, [0, 1])
    assert epochs.trials[0]["id"] == "sub-001_run-4_sample-160"
    assert epochs.trials[0]["start_sample"] == 320
    assert epochs.trials[0]["stop_sample"] == 800
    assert epochs.trials[0]["onset_s"] == 1
    np.testing.assert_array_equal(values, original)
    assert not epochs.X.flags.writeable
    assert not epochs.y.flags.writeable


@pytest.mark.parametrize(
    "event,match",
    [
        ({"sample": 160, "label": "TASK2T1", "duration_s": 3.9}, "annotated imagery"),
        ({"sample": 1800, "label": "TASK2T2", "duration_s": 4.1}, "recording end"),
        ({"sample": 0, "label": "TASK2T1", "duration_s": 4.1}, "warmup"),
        ({"sample": 160, "label": "TASK1T1", "duration_s": 4.1}, "Unexpected event label"),
    ],
)
def test_invalid_epoch_is_explicit_failure_not_silent_rejection(event, match):
    with pytest.raises(ValueError, match=match):
        _extract_epochs(np.zeros((3, 2000)), [event], ("C3", "Cz", "C4"), 160, 4, CleanConfig())


def test_sample_conversion_rejects_fractional_samples():
    assert _sample(4.2, 160) == 672
    with pytest.raises(ValueError, match="align with a sample"):
        _sample(4.201, 160)


def test_checksum_validation_detects_modified_source(tmp_path):
    path = tmp_path / "source.set"
    path.write_bytes(b"audited")
    entries = [{"path": path.name, "sha256": hashlib.sha256(b"audited").hexdigest(), "size": 7}]
    assert _verify_files(tmp_path, entries) == entries
    path.write_bytes(b"changed")
    with pytest.raises(ValueError, match="checksum or size mismatch"):
        _verify_files(tmp_path, entries)


@pytest.mark.integration
def test_cached_audited_data_units_events_split_and_continuous_preprocessing():
    root = Path(__file__).resolve().parents[1]
    data_root = root / "data/ds004362"
    if not (data_root / "sub-001/eeg/sub-001_task-motion_run-4_eeg.set").is_file():
        pytest.skip("Audited dataset is not cached; this test never downloads data")
    prepared = prepare_data(data_root, root / "configs/sub-001-manifest.json", CleanConfig())
    assert prepared.train.X.shape == (30, 64, 480)
    assert prepared.test.X.shape == (15, 64, 480)
    np.testing.assert_array_equal(np.bincount(prepared.train.y), [16, 14])
    np.testing.assert_array_equal(np.bincount(prepared.test.y), [7, 8])
    assert prepared.train.channels == prepared.test.channels
    assert {trial["run"] for trial in prepared.train.trials} == {4, 8}
    assert {trial["run"] for trial in prepared.test.trials} == {12}
    assert not ({trial["id"] for trial in prepared.train.trials}
                & {trial["id"] for trial in prepared.test.trials})
    assert prepared.train.trials[0]["start_sample"] == 832
    assert prepared.train.trials[0]["stop_sample"] == 1312
    assert prepared.test.trials[-1]["stop_sample"] == 19904
    assert prepared.audit["source_files_unchanged"] is True
    for run in prepared.audit["runs"]:
        assert run["edf_set_samples_equal_in_volts"] is True
        assert run["exclusions"] == []
        assert run["duration_s"] == 125
        assert run["events"][-1]["duration_s"] == pytest.approx(4.1)
        assert run["unannotated_tail_s"] == 0.5
    from scipy.io import loadmat

    source = loadmat(
        data_root / "sub-001/eeg/sub-001_task-motion_run-4_eeg.set", simplify_cells=True
    )["data"].astype(float) * 1e-6
    expected = sosfilt(butter(4, [8, 30], btype="bandpass", fs=160, output="sos"), source)
    np.testing.assert_array_equal(prepared.train.X[0], expected[:, 832:1312])
