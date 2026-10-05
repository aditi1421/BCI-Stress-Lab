"""Deterministic corruption of continuous EEG in its supplied native units.

Inputs and outputs are (channels, samples). No unit conversion, centering,
filtering, reference changes, or learned statistics occur in these operators.
For the audited SET recordings values and training RMS are in microvolts.
"""

from collections.abc import Sequence
from numbers import Integral, Real

import numpy as np


def _continuous(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values)
    if array.dtype.kind not in "iuf":
        raise ValueError("Continuous EEG must contain real numeric values")
    array = np.asarray(array, dtype=np.float64)
    if array.ndim != 2 or any(size == 0 for size in array.shape):
        raise ValueError("Expected nonempty continuous (channels, samples) EEG")
    if not np.isfinite(array).all():
        raise ValueError("Continuous EEG must contain finite values")
    return array


def _seed(seed: int) -> int:
    if isinstance(seed, bool) or not isinstance(seed, Integral) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    return int(seed)


def _names(names: Sequence[str], description: str) -> tuple[str, ...]:
    if isinstance(names, str):
        raise TypeError(f"{description} must be a sequence of channel names")
    result = tuple(names)
    if any(not isinstance(name, str) or not name for name in result):
        raise ValueError(f"{description} must contain nonempty strings")
    if len(set(result)) != len(result):
        raise ValueError(f"{description} must not contain duplicate names")
    return result


def training_rms(training_runs: Sequence[np.ndarray]) -> np.ndarray:
    """Return pooled, uncentered RMS per channel across training runs only.

    RMS[channel] = sqrt(sum of squared samples / total training samples).
    Runs may have different lengths, but must use identical channel order.
    Callers own the training split and channel-order validation: this function
    receives no labels, run identifiers, or held-out recordings.
    """
    if len(training_runs) == 0:
        raise ValueError("At least one training run is required")
    arrays = [_continuous(run) for run in training_runs]
    channel_count = arrays[0].shape[0]
    if any(array.shape[0] != channel_count for array in arrays):
        raise ValueError("Training runs must have the same channel count and order")
    total_samples = sum(array.shape[1] for array in arrays)
    sum_squares = np.zeros(channel_count, dtype=np.float64)
    with np.errstate(over="ignore", invalid="ignore"):
        for array in arrays:
            sum_squares += np.sum(np.square(array), axis=1)
        scale = np.sqrt(sum_squares / total_samples)
    if not np.isfinite(scale).all():
        raise ValueError("Training RMS overflowed; check native signal units")
    return scale


def add_gaussian_noise(
    values: np.ndarray, scale: np.ndarray, level: float, seed: int,
) -> np.ndarray:
    """Add independent Gaussian noise with SD = level * training RMS[channel].

    ``scale`` must be in the same native units as ``values``. It is computed
    from training recordings before filtering, never from this test recording.
    Reusing a seed reuses the standard-normal realization at every severity.
    A zero level returns an independent, numerically identical float64 copy.
    """
    array = _continuous(values)
    channel_scale = np.asarray(scale)
    if channel_scale.dtype.kind not in "iuf":
        raise ValueError("scale must contain real numeric values")
    channel_scale = np.asarray(channel_scale, dtype=np.float64)
    if channel_scale.shape != (array.shape[0],):
        raise ValueError("scale must have one RMS value per channel")
    if not np.isfinite(channel_scale).all() or np.any(channel_scale < 0):
        raise ValueError("scale must contain finite, nonnegative RMS values")
    if isinstance(level, bool) or not isinstance(level, Real):
        raise TypeError("level must be a finite, nonnegative number")
    if not np.isfinite(level) or level < 0:
        raise ValueError("level must be a finite, nonnegative number")
    rng = np.random.default_rng(_seed(seed))
    if level == 0:
        return array.copy()
    with np.errstate(over="ignore", invalid="ignore"):
        result = array + (float(level) * channel_scale[:, None]) * rng.standard_normal(
            array.shape,
        )
    if not np.isfinite(result).all():
        raise ValueError("Corrupted EEG overflowed; check level and native signal units")
    return result


def flatline_channels(
    values: np.ndarray,
    channel_names: Sequence[str],
    *,
    count: int | None = None,
    names: Sequence[str] | None = None,
    seed: int = 0,
) -> tuple[np.ndarray, tuple[str, ...]]:
    """Set selected channels to zero for the full continuous recording.

    Supply exactly one of ``count`` or ``names``. Random selection uses a
    seeded permutation: larger counts with the same seed contain all channels
    selected at smaller counts. Explicit names retain their supplied order.
    Returns a new float64 array and the exact selected names. Channel positions
    and the recorded reference are preserved; there is no rereferencing.
    """
    array = _continuous(values)
    all_names = _names(channel_names, "channel_names")
    if len(all_names) != array.shape[0]:
        raise ValueError("channel_names must match the continuous EEG channel count")
    random_seed = _seed(seed)
    if (count is None) == (names is None):
        raise ValueError("Supply exactly one of count or names")
    if names is not None:
        selected_names = _names(names, "names")
        unknown = set(selected_names) - set(all_names)
        if unknown:
            raise ValueError(f"Unknown channel names: {sorted(unknown)}")
        selected = np.asarray([all_names.index(name) for name in selected_names], dtype=int)
    else:
        if isinstance(count, bool) or not isinstance(count, Integral):
            raise ValueError("count must be an integer")
        if count < 0 or count > len(all_names):
            raise ValueError("count must be between zero and the number of channels")
        selected = np.random.default_rng(random_seed).permutation(len(all_names))[:count]
        selected_names = tuple(all_names[index] for index in selected)
    result = array.copy()
    result[selected] = 0.0
    return result, selected_names
