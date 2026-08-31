from __future__ import annotations

import csv
import os
from pathlib import Path
from typing import Iterable

import pandas as pd


DEFAULT_DATASET_NAME = "ncc_dataset.csv"


def _iter_candidate_paths(path: str | os.PathLike[str] | None = None) -> Iterable[Path]:
    if path is not None:
        yield Path(path).expanduser().resolve()

    project_root = Path(__file__).resolve().parent.parent
    yield project_root / "data" / DEFAULT_DATASET_NAME
    yield project_root / "notebook_project" / "data" / DEFAULT_DATASET_NAME
    yield project_root / DEFAULT_DATASET_NAME
    yield Path.cwd() / "data" / DEFAULT_DATASET_NAME
    yield Path.cwd() / DEFAULT_DATASET_NAME

    env_path = os.environ.get("NCC_DATASET_PATH")
    if env_path:
        yield Path(env_path).expanduser().resolve()


def _build_default_dataset(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        {"sample_id": 1, "feature_a": 0.12, "feature_b": 1.10, "label": "A"},
        {"sample_id": 2, "feature_a": 0.87, "feature_b": 2.30, "label": "B"},
        {"sample_id": 3, "feature_a": 1.12, "feature_b": 1.50, "label": "A"},
        {"sample_id": 4, "feature_a": 2.35, "feature_b": 3.10, "label": "B"},
        {"sample_id": 5, "feature_a": 0.65, "feature_b": 1.80, "label": "A"},
        {"sample_id": 6, "feature_a": 1.75, "feature_b": 2.10, "label": "B"},
        {"sample_id": 7, "feature_a": 1.95, "feature_b": 2.90, "label": "B"},
        {"sample_id": 8, "feature_a": 0.40, "feature_b": 1.20, "label": "A"},
        {"sample_id": 9, "feature_a": 2.70, "feature_b": 3.40, "label": "B"},
        {"sample_id": 10, "feature_a": 0.95, "feature_b": 1.60, "label": "A"},
    ]
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=["sample_id", "feature_a", "feature_b", "label"])
        writer.writeheader()
        writer.writerows(rows)


def load_ncc_dataset(path: str | os.PathLike[str] | None = None) -> pd.DataFrame:
    """Load the NCC dataset from disk, creating a default sample CSV when absent.

    The loader searches a few common project-relative paths so it works both from the
    repository root and from notebook execution contexts.
    """
    candidate_paths = list(_iter_candidate_paths(path))
    resolved_path: Path | None = None

    for candidate in candidate_paths:
        if candidate.exists():
            resolved_path = candidate
            break

    if resolved_path is None:
        resolved_path = candidate_paths[0] if candidate_paths else Path.cwd() / "data" / DEFAULT_DATASET_NAME
        _build_default_dataset(resolved_path)

    return pd.read_csv(resolved_path)
