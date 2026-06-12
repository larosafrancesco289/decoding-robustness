"""Load and validate experiment configs from YAML."""

from __future__ import annotations

from pathlib import Path

import yaml

from .schema import ExperimentConfig


def load_experiment_config(path: str | Path) -> ExperimentConfig:
    """Parse a YAML experiment config and validate it against the schema.

    Raises pydantic.ValidationError on a malformed config (e.g. a decoding method
    that sets a parameter it shouldn't), and ValueError if the file is not a mapping.
    """
    path = Path(path)
    with path.open(encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: top-level YAML must be a mapping, got {type(raw).__name__}")
    return ExperimentConfig.model_validate(raw)
