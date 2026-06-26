"""YAML configuration loading with shallow recursive inheritance."""

from pathlib import Path
from typing import Any, Dict

import yaml


def _deep_update(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            base[key] = _deep_update(base[key], value)
        else:
            base[key] = value
    return base


def load_config(path: str) -> Dict[str, Any]:
    """Load a YAML config and merge an optional ``extends`` parent first."""
    path_obj = Path(path)
    with path_obj.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    parent = cfg.pop("extends", None)
    if parent:
        parent_path = Path(parent)
        if not parent_path.is_absolute():
            parent_path = path_obj.parent.parent / parent_path
        base = load_config(str(parent_path))
        return _deep_update(base, cfg)
    return cfg

