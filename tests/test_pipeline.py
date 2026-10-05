"""Synthetic checks of feature definitions and fixed-decoder isolation."""

import numpy as np
import pytest
from sklearn.base import clone

from bci_stress.config import CleanConfig
from bci_stress.pipeline import LogBandpower, make_pipelines


@pytest.fixture
def synthetic_epochs():
    rng = np.random.default_rng(713)
    labels = np.tile([0, 1], 16)
    epochs = rng.normal(size=(32, 6, 128)) * 1e-6
    epochs[labels == 0, :2] *= 3
    epochs[labels == 1, 2:4] *= 3
    return epochs[:24], labels[:24], epochs[24:], labels[24:]


def test_log_bandpower_exact_definition_and_stateless_fit():
    values = np.array([[[1., 2., 3.], [0., 0., 0.]]])
    transformer = LogBandpower(power_floor=1e-12)
    state = vars(transformer).copy()
    expected = np.log([[14 / 3, 1e-12]])
    np.testing.assert_allclose(transformer.transform(values), expected)
    assert transformer.fit(values) is transformer
    assert vars(transformer) == state
    np.testing.assert_allclose(transformer.transform(values), expected)
    # Nonzero mean is deliberately retained; mean squared signal != variance.
    assert transformer.transform(np.ones((1, 1, 3)))[0, 0] == 0
    assert clone(transformer).get_params() == transformer.get_params()


@pytest.mark.parametrize("values", [
    np.ones((2, 3)), np.ones((0, 3, 4)), np.ones((2, 0, 4)),
    np.ones((2, 3, 0)), np.full((1, 1, 2), np.nan),
    np.full((1, 1, 2), np.inf),
])
def test_log_bandpower_rejects_invalid_epochs(values):
    with pytest.raises(ValueError):
        LogBandpower().fit(values)
    with pytest.raises(ValueError):
        LogBandpower().transform(values)


@pytest.mark.parametrize("floor", [0, -1, np.nan, np.inf])
def test_log_bandpower_rejects_invalid_floor(floor):
    with pytest.raises(ValueError, match="power_floor"):
        LogBandpower(floor).fit(np.ones((1, 1, 2)))


def test_pipelines_share_classifier_settings_and_not_fitted_objects():
    pipelines = make_pipelines(CleanConfig())
    assert set(pipelines) == {"csp_lda", "bandpower_lda"}
    csp_lda = pipelines["csp_lda"].named_steps["lda"]
    bandpower_lda = pipelines["bandpower_lda"].named_steps["lda"]
    assert csp_lda is not bandpower_lda
    assert csp_lda.get_params() == bandpower_lda.get_params()
    assert csp_lda.solver == "lsqr"
    assert csp_lda.shrinkage == "auto"
    assert csp_lda.priors == [0.5, 0.5]
    for pipeline in pipelines.values():
        assert not hasattr(pipeline.named_steps["lda"], "coef_")


def test_csp_features_are_log_mean_squared_fixed_projections(synthetic_epochs):
    train, labels, heldout, _ = synthetic_epochs
    pipeline = make_pipelines(CleanConfig())["csp_lda"].fit(train, labels)
    csp = pipeline.named_steps["csp"]
    assert csp.n_components == 4
    assert csp.reg == 0.1
    assert csp.cov_est == "concat"
    assert csp.rank == "full"
    assert not csp.norm_trace
    assert csp.transform_into == "csp_space"
    assert csp.log is None
    projections = np.einsum("kc,nct->nkt", csp.filters_[:4], heldout)
    expected = np.log(np.maximum(np.mean(projections ** 2, axis=-1), 1e-24))
    np.testing.assert_allclose(pipeline[:-1].transform(heldout), expected, atol=1e-12)
    csp.mean_[:] = 1e30
    csp.std_[:] = 1e-30
    np.testing.assert_allclose(pipeline[:-1].transform(heldout), expected, atol=1e-12)


@pytest.mark.parametrize("name", ["csp_lda", "bandpower_lda"])
def test_fitted_pipeline_clones_without_learned_state(name, synthetic_epochs):
    train, labels, _, _ = synthetic_epochs
    fitted = make_pipelines(CleanConfig())[name].fit(train, labels)
    cloned = clone(fitted)
    fresh = make_pipelines(CleanConfig())[name]
    for step_name, fitted_step in fitted.named_steps.items():
        assert cloned.named_steps[step_name] is not fitted_step
        assert fresh.named_steps[step_name] is not fitted_step
        assert cloned.named_steps[step_name].get_params() == fitted_step.get_params()
        assert not hasattr(cloned.named_steps[step_name], "coef_")
        assert not hasattr(cloned.named_steps[step_name], "filters_")
        assert not hasattr(fresh.named_steps[step_name], "coef_")
        assert not hasattr(fresh.named_steps[step_name], "filters_")


@pytest.mark.parametrize("name", ["csp_lda", "bandpower_lda"])
def test_synthetic_power_signal_decodes_but_permuted_training_labels_do_not(name):
    rng = np.random.default_rng(1873)
    labels = np.tile([0, 1], 80)
    epochs = rng.normal(size=(160, 6, 128)) * 1e-6
    epochs[labels == 0, :2] *= 3
    epochs[labels == 1, 2:4] *= 3
    train, heldout = epochs[:32], epochs[32:]
    train_labels, heldout_labels = labels[:32], labels[32:]
    pipeline = make_pipelines(CleanConfig())[name]
    assert pipeline.fit(train, train_labels).score(heldout, heldout_labels) > 0.95
    permuted_scores = []
    for _ in range(32):
        null_pipeline = clone(pipeline)
        null_pipeline.fit(train, rng.permutation(train_labels))
        permuted_scores.append(null_pipeline.score(heldout, heldout_labels))
    assert 0.35 < np.mean(permuted_scores) < 0.65


@pytest.mark.parametrize("name", ["csp_lda", "bandpower_lda"])
def test_prediction_is_batch_independent_and_does_not_fit_or_mutate(
    name, synthetic_epochs, monkeypatch,
):
    train, labels, heldout, _ = synthetic_epochs
    train_before = train.copy()
    heldout_before = heldout.copy()
    pipeline = make_pipelines(CleanConfig())[name]
    pipeline.fit(train, labels)
    coefficients = pipeline.named_steps["lda"].coef_.copy()
    filters = pipeline.named_steps["csp"].filters_.copy() if name == "csp_lda" else None

    def fail_fit(*args, **kwargs):
        pytest.fail("Prediction attempted to fit a transformer/classifier")

    for step in pipeline.named_steps.values():
        monkeypatch.setattr(step, "fit", fail_fit)
    batch = pipeline.predict_proba(heldout)
    separate = np.concatenate([pipeline.predict_proba(epoch[None]) for epoch in heldout])
    np.testing.assert_allclose(batch, separate, atol=1e-12)
    np.testing.assert_array_equal(pipeline.named_steps["lda"].coef_, coefficients)
    if filters is not None:
        np.testing.assert_array_equal(pipeline.named_steps["csp"].filters_, filters)
    np.testing.assert_array_equal(train, train_before)
    np.testing.assert_array_equal(heldout, heldout_before)
