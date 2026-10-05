"""Two fixed decoders with shared log-power features and shrinkage LDA."""

import numpy as np
from mne.decoding import CSP
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.pipeline import Pipeline

from .config import CleanConfig


class LogBandpower(TransformerMixin, BaseEstimator):
    """Natural log of mean squared band-filtered signal, with no fitted statistics.

    The input is (epochs, channels or CSP components, time samples). Filtering
    happens upstream, identically for both decoders. No sample/channel centering
    is applied here: these are power features, not sample variances.

    The numerical floor is in squared input units: V² for sensor features, or
    squared CSP projection units after covariance-normalized spatial filtering.
    """

    def __init__(self, power_floor: float = 1e-24):
        self.power_floor = power_floor

    def _validate_input(self, X: np.ndarray) -> np.ndarray:
        if not np.isfinite(self.power_floor) or self.power_floor <= 0:
            raise ValueError("power_floor must be finite and positive")
        values = np.asarray(X, dtype=np.float64)
        if values.ndim != 3 or any(size == 0 for size in values.shape):
            raise ValueError("Expected nonempty (epochs, channels, samples) input")
        if not np.isfinite(values).all():
            raise ValueError("Epoch values must be finite")
        return values

    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> "LogBandpower":
        self._validate_input(X)
        return self

    def __sklearn_is_fitted__(self) -> bool:
        return True

    def transform(self, X: np.ndarray) -> np.ndarray:
        values = self._validate_input(X)
        with np.errstate(over="ignore"):
            power = np.mean(np.square(values), axis=-1)
        if not np.isfinite(power).all():
            raise ValueError("Signal power overflowed; check signal units")
        return np.log(np.maximum(power, self.power_floor))


def make_pipelines(config: CleanConfig) -> dict[str, Pipeline]:
    """Create fresh, unfitted pipelines; call fit with training epochs only."""
    csp = CSP(
        n_components=config.csp_components,
        reg=config.csp_regularization,
        cov_est="concat",
        rank="full",
        norm_trace=False,
        component_order="mutual_info",
        transform_into="csp_space",
        log=None,
    )
    pipelines = {}
    for name, spatial_steps in (("csp_lda", [("csp", csp)]), ("bandpower_lda", [])):
        pipelines[name] = Pipeline([
            *spatial_steps,
            ("log_bandpower", LogBandpower(power_floor=config.power_floor)),
            ("lda", LinearDiscriminantAnalysis(
                solver="lsqr", shrinkage="auto", priors=[0.5, 0.5],
            )),
        ])
    return pipelines
