"""Explicit, validated configuration for the audited clean-decoding experiment."""

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

DATASET_COMMIT = "af0c461f23e8a7aa782475e3c22d160bc39eb170"


class CleanConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    subject: Literal[1] = 1
    dataset_commit: Literal["af0c461f23e8a7aa782475e3c22d160bc39eb170"] = DATASET_COMMIT
    train_runs: tuple[Literal[4], Literal[8]] = (4, 8)
    test_run: Literal[12] = 12
    epoch_start_s: Literal[1.0] = 1.0
    epoch_stop_s: Literal[4.0] = 4.0
    low_hz: float = Field(default=8.0, gt=0)
    high_hz: float = Field(default=30.0, gt=0, lt=80)
    filter_order: int = Field(default=4, ge=1, le=8)
    warmup_s: float = Field(default=2.0, ge=0, le=5.2)
    csp_components: int = Field(default=4, ge=1, le=64)
    csp_regularization: float = Field(default=0.1, gt=0, lt=1)
    power_floor: float = Field(default=1e-24, gt=0)
    permutation_count: int = Field(default=199, ge=1)
    permutation_seed: int = Field(default=20261005, ge=0)

    @model_validator(mode="after")
    def valid_band(self):
        if self.low_hz >= self.high_hz:
            raise ValueError("Bandpass lower frequency must be below upper frequency")
        return self

    @classmethod
    def load(cls, path: Path) -> "CleanConfig":
        return cls.model_validate_json(path.read_text())
