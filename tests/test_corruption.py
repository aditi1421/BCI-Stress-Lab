"""Native-unit, reproducibility, and input-isolation checks for EEG corruption."""

import numpy as np
import pytest

from bci_stress.corruption import add_gaussian_noise, flatline_channels, training_rms


@pytest.fixture
def continuous_eeg():
    return np.random.default_rng(101).normal(size=(6, 2048)) * 20.0


def test_training_rms_pools_samples_without_centering_or_mutation():
    first = np.array([[3., 3.], [0., 4.]])
    second = np.array([[4., 4., 4., 4.], [2., 2., 2., 2.]])
    originals = [first.copy(), second.copy()]
    scale = training_rms([first, second])
    np.testing.assert_allclose(scale, np.sqrt([82 / 6, 32 / 6]), rtol=1e-15)
    for actual, original in zip([first, second], originals, strict=True):
        np.testing.assert_array_equal(actual, original)
    np.testing.assert_array_equal(training_rms([np.full((2, 4), -3.0)]), [3., 3.])
    np.testing.assert_array_equal(training_rms([np.zeros((2, 4))]), [0., 0.])


def test_gaussian_noise_exact_generator_and_per_channel_scale(continuous_eeg):
    scale = np.array([0., 1., 3., 10., 20., 50.])
    original = continuous_eeg.copy()
    original_scale = scale.copy()
    noise = np.random.default_rng(71).standard_normal(continuous_eeg.shape)
    expected = continuous_eeg + 0.5 * scale[:, None] * noise
    actual = add_gaussian_noise(continuous_eeg, scale, level=0.5, seed=71)
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(actual[0], continuous_eeg[0])
    np.testing.assert_array_equal(continuous_eeg, original)
    np.testing.assert_array_equal(scale, original_scale)
    assert actual.shape == continuous_eeg.shape
    assert actual.dtype == np.float64
    assert not np.shares_memory(actual, continuous_eeg)


def test_gaussian_noise_has_expected_empirical_mean_and_standard_deviation():
    values = np.zeros((3, 100_000))
    scale = np.array([2., 20., 200.])
    result = add_gaussian_noise(values, scale, level=0.5, seed=910)
    standardized = result / (0.5 * scale[:, None])
    np.testing.assert_allclose(standardized.mean(axis=1), 0, atol=0.01)
    np.testing.assert_allclose(standardized.std(axis=1), 1, atol=0.01)
    correlation = np.corrcoef(standardized)
    np.testing.assert_allclose(correlation, np.eye(3), atol=0.01)


def test_gaussian_noise_seed_repeatability_and_linear_severity(continuous_eeg):
    scale = training_rms([continuous_eeg])
    smaller = add_gaussian_noise(continuous_eeg, scale, level=0.25, seed=81)
    repeated = add_gaussian_noise(continuous_eeg, scale, level=0.25, seed=81)
    larger = add_gaussian_noise(continuous_eeg, scale, level=1, seed=81)
    different = add_gaussian_noise(continuous_eeg, scale, level=0.25, seed=82)
    np.testing.assert_array_equal(smaller, repeated)
    np.testing.assert_allclose(larger - continuous_eeg, 4 * (smaller - continuous_eeg),
                               rtol=1e-12, atol=1e-13)
    assert not np.array_equal(smaller, different)


def test_native_units_are_not_implicitly_converted(continuous_eeg):
    scale = training_rms([continuous_eeg])
    microvolts = add_gaussian_noise(continuous_eeg, scale, level=0.5, seed=310)
    volts = add_gaussian_noise(continuous_eeg * 1e-6, scale * 1e-6, level=0.5, seed=310)
    np.testing.assert_allclose(volts, microvolts * 1e-6, rtol=1e-12, atol=1e-20)
    np.testing.assert_allclose(training_rms([continuous_eeg * 1e-6]), scale * 1e-6)


def test_zero_severity_is_exact_independent_copy(continuous_eeg):
    names = tuple(f"channel_{index}" for index in range(6))
    original = continuous_eeg.copy()
    continuous_eeg.setflags(write=False)
    outputs = [add_gaussian_noise(continuous_eeg, np.ones(6), level=0, seed=9)]
    for selection in ({"count": 0}, {"names": ()}):
        result, failed = flatline_channels(continuous_eeg, names, **selection)
        assert failed == ()
        outputs.append(result)
    for result in outputs:
        np.testing.assert_array_equal(result, original)
        assert not np.shares_memory(result, continuous_eeg)
        result[0, 0] = 1000
    np.testing.assert_array_equal(continuous_eeg, original)


def test_flatline_nested_channel_selection_and_source_isolation(continuous_eeg):
    names = tuple(f"channel_{index}" for index in range(6))
    original = continuous_eeg.copy()
    expected_order = np.random.default_rng(19).permutation(6)
    previous_names = ()
    continuous_eeg.setflags(write=False)
    for count in [0, 1, 2, 4, 6]:
        result, failed = flatline_channels(continuous_eeg, names, count=count, seed=19)
        assert failed == tuple(names[index] for index in expected_order[:count])
        assert failed[:len(previous_names)] == previous_names
        assert len(failed) == count
        assert result.shape == continuous_eeg.shape
        assert result.dtype == np.float64
        assert not np.shares_memory(result, continuous_eeg)
        np.testing.assert_array_equal(result[expected_order[:count]], 0)
        np.testing.assert_array_equal(result[expected_order[count:]], original[expected_order[count:]])
        repeated, repeated_names = flatline_channels(continuous_eeg, names, count=count, seed=19)
        np.testing.assert_array_equal(result, repeated)
        assert failed == repeated_names
        previous_names = failed
    np.testing.assert_array_equal(continuous_eeg, original)


def test_flatline_explicit_names_preserve_order_and_all_other_values():
    values = np.arange(32).reshape(4, 8)
    names = ["C3", "C4", "Cz", "Fz"]
    result, failed = flatline_channels(values, names, names=["Fz", "C3"], seed=99)
    assert failed == ("Fz", "C3")
    assert result.dtype == np.float64
    np.testing.assert_array_equal(result[[3, 0]], 0)
    np.testing.assert_array_equal(result[[1, 2]], values[[1, 2]])
    np.testing.assert_array_equal(values, np.arange(32).reshape(4, 8))
    other_seed, same_names = flatline_channels(values, names, names=["Fz", "C3"], seed=1)
    np.testing.assert_array_equal(result, other_seed)
    assert failed == same_names


def test_local_generators_do_not_use_global_random_state(monkeypatch, continuous_eeg):
    def fail_global_rng(*args, **kwargs):
        pytest.fail("Corruption used the global NumPy random generator")

    for method in ["seed", "normal", "standard_normal", "permutation"]:
        monkeypatch.setattr(np.random, method, fail_global_rng)
    add_gaussian_noise(continuous_eeg, np.ones(6), level=1, seed=31)
    flatline_channels(continuous_eeg, [str(index) for index in range(6)], count=2, seed=31)


@pytest.mark.parametrize("values", [
    np.ones(8), np.ones((2, 3, 4)), np.ones((0, 8)), np.ones((2, 0)),
    np.full((2, 8), np.nan), np.full((2, 8), np.inf),
    np.full((2, 8), 1 + 1j), np.full((2, 8), "1"), np.ones((2, 8), dtype=bool),
])
def test_all_operators_reject_invalid_continuous_data(values):
    with pytest.raises(ValueError):
        training_rms([values])
    with pytest.raises(ValueError):
        add_gaussian_noise(values, np.ones(2), level=0, seed=1)
    with pytest.raises(ValueError):
        flatline_channels(values, ["C3", "C4"], count=0)


@pytest.mark.parametrize("runs", [[], [np.ones((2, 8)), np.ones((3, 8))]])
def test_training_rms_rejects_empty_or_mismatched_runs(runs):
    with pytest.raises(ValueError):
        training_rms(runs)


@pytest.mark.parametrize("scale", [
    1, np.ones((2, 1)), np.ones(3), [-1, 2], [np.nan, 2], [np.inf, 2],
    [1 + 1j, 2], ["1", "2"], [True, False],
])
def test_gaussian_rejects_invalid_scale_even_at_zero_severity(scale):
    with pytest.raises(ValueError, match="scale"):
        add_gaussian_noise(np.ones((2, 8)), scale, level=0, seed=1)


@pytest.mark.parametrize("level", [-1, np.nan, np.inf, True, "1", 1 + 1j])
def test_gaussian_rejects_invalid_severity(level):
    with pytest.raises((TypeError, ValueError), match="level"):
        add_gaussian_noise(np.ones((2, 8)), np.ones(2), level=level, seed=1)


@pytest.mark.parametrize("seed", [-1, 1.5, "1", True, None])
def test_operators_reject_invalid_seed_even_at_zero_severity(seed):
    with pytest.raises(ValueError, match="seed"):
        add_gaussian_noise(np.ones((2, 8)), np.ones(2), level=0, seed=seed)
    with pytest.raises(ValueError, match="seed"):
        flatline_channels(np.ones((2, 8)), ["C3", "C4"], count=0, seed=seed)


@pytest.mark.parametrize("selection", [
    {}, {"count": 1, "names": ["C3"]}, {"count": -1}, {"count": 3},
    {"count": 0.5}, {"count": True}, {"names": ["unknown"]},
    {"names": ["C3", "C3"]}, {"names": "C3"}, {"names": [1]}, {"names": [""]},
])
def test_flatline_rejects_ambiguous_or_invalid_selection(selection):
    with pytest.raises((TypeError, ValueError)):
        flatline_channels(np.ones((2, 8)), ["C3", "C4"], **selection)


@pytest.mark.parametrize("names", [
    ["C3"], ["C3", "C3"], ["C3", ""], ["C3", 1], "C3",
])
def test_flatline_rejects_invalid_channel_names(names):
    with pytest.raises((TypeError, ValueError), match="channel_names"):
        flatline_channels(np.ones((2, 8)), names, count=0)


def test_integer_numpy_seed_and_count_supported():
    values = np.ones((2, 8))
    add_gaussian_noise(values, np.ones(2), level=np.float64(0.5), seed=np.int64(3))
    _, names = flatline_channels(values, ["C3", "C4"], count=np.int64(1), seed=np.int64(3))
    assert len(names) == 1


def test_overflow_reports_bad_units_instead_of_returning_nonfinite_values():
    with pytest.raises(ValueError, match="overflowed"):
        training_rms([np.full((2, 8), 1e308)])
    with pytest.raises(ValueError, match="overflowed"):
        add_gaussian_noise(np.ones((2, 8)), np.full(2, 1e308), level=1e308, seed=1)
