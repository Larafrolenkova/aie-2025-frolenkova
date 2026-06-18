from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_config(config_path: str | Path | None = None) -> dict[str, Any]:
    """Load YAML configuration and resolve project-relative paths."""
    raw_path = config_path or os.getenv("CONFIG_PATH", "configs/config.yaml")
    path = Path(raw_path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path

    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")

    with path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    for key, value in config.get("paths", {}).items():
        value_path = Path(value)
        if not value_path.is_absolute():
            config["paths"][key] = str(PROJECT_ROOT / value_path)

    return config
