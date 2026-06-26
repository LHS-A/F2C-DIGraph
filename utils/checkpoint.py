"""Checkpoint save/load helpers."""

from pathlib import Path
from typing import Any, Dict, Optional

import torch


def save_checkpoint(path: str, state: Dict[str, Any]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)


def load_checkpoint(path: str, map_location: Optional[str] = "cpu") -> Dict[str, Any]:
    if not Path(path).exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    return torch.load(path, map_location=map_location)

