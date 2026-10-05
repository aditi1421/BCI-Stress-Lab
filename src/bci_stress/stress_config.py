"""Explicit settings for the first frozen-decoder corruption experiment."""

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class StressConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    noise_levels: list[float] = Field(default_factory=lambda: [0, 0.25, 0.5, 1, 2], min_length=1)
    dropout_counts: list[int] = Field(default_factory=lambda: [0, 1, 2, 4, 8, 16], min_length=1)
    seeds: list[int] = Field(default_factory=lambda: list(range(10)), min_length=2)
    noise_scale: Literal["per_channel_training_continuous_rms"] = "per_channel_training_continuous_rms"
    native_units: Literal["uV"] = "uV"

    @model_validator(mode="after")
    def validate_grid(self):
        for grid in (self.noise_levels, self.dropout_counts):
            if grid[0] != 0 or any(value < 0 for value in grid) or sorted(set(grid)) != grid:
                raise ValueError("Severity grids must start at zero and increase without duplicates")
        if max(self.dropout_counts) > 64:
            raise ValueError("Cannot flatline more than the 64 recorded channels")
        if min(self.seeds) < 0 or len(set(self.seeds)) != len(self.seeds):
            raise ValueError("Seeds must be unique nonnegative integers")
        return self

    @classmethod
    def load(cls, path: Path) -> "StressConfig":
        return cls.model_validate_json(path.read_text())
