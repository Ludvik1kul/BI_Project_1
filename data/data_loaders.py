from __future__ import annotations

from pathlib import Path

import pandas as pd

try:
    from datasets import load_dataset, load_from_disk
except Exception:  # pragma: no cover - fallback for Python versions with incompatible datasets releases
    load_dataset = None
    load_from_disk = None

from huggingface_hub import hf_hub_download, list_repo_files

DEFAULT_DATASET_NAME = "NbAiLab/NCC"
LOCAL_NCC_PATH = Path(__file__).resolve().parent / "ncc"


def _load_local_jsonl_dataset(dataset_dir: Path, max_rows: int | None = None) -> pd.DataFrame:
    """Load raw NCC JSONL shards already saved in the local data/ncc folder and merge them."""
    shard_dir = dataset_dir / "data"
    jsonl_files = sorted(shard_dir.glob("*.jsonl")) if shard_dir.exists() else []
    if not jsonl_files:
        raise FileNotFoundError(f"No NCC JSONL shards were found in {shard_dir}.")

    frames = [pd.read_json(path, lines=True) for path in jsonl_files]
    dataset = pd.concat(frames, ignore_index=True)
    if max_rows is not None:
        dataset = dataset.head(max_rows)
    dataset_path = dataset_dir / "ncc_dataset.csv"
    dataset.to_csv(dataset_path, index=False)
    return dataset


def _load_ncc_as_dataframe(dataset_dir: Path, max_rows: int | None = None) -> pd.DataFrame:
    """Load the NCC JSONL shards from Hugging Face and merge them into a single DataFrame."""
    repo_files = list_repo_files(repo_id=DEFAULT_DATASET_NAME, repo_type="dataset", revision="main")
    matched_files = sorted(path for path in repo_files if path.startswith("data/") and path.endswith(".jsonl"))

    frames = []
    for relative_path in matched_files:
        local_file = hf_hub_download(
            repo_id=DEFAULT_DATASET_NAME,
            repo_type="dataset",
            filename=relative_path,
            local_dir=str(dataset_dir),
        )
        frames.append(pd.read_json(local_file, lines=True))

    dataset = pd.concat(frames, ignore_index=True)
    if max_rows is not None:
        dataset = dataset.head(max_rows)
    dataset_path = dataset_dir / "ncc_dataset.csv"
    dataset.to_csv(dataset_path, index=False)
    return dataset


def load_ncc_dataset(
    dataset_name: str = DEFAULT_DATASET_NAME,
    local_path: str | Path | None = None,
    max_rows: int | None = None,
):
    """Load the NCC dataset locally when available, otherwise download it from Hugging Face and save it under data/ncc."""
    dataset_dir = Path(local_path) if local_path is not None else LOCAL_NCC_PATH
    dataset_dir.mkdir(parents=True, exist_ok=True)

    local_csv = dataset_dir / "ncc_dataset.csv"
    if local_csv.exists():
        dataset = pd.read_csv(local_csv)
        if max_rows is not None:
            return dataset.head(max_rows)
        return dataset

    saved_jsonl_dir = dataset_dir / "data"
    if saved_jsonl_dir.exists() and any(saved_jsonl_dir.glob("*.jsonl")):
        return _load_local_jsonl_dataset(dataset_dir, max_rows=max_rows)

    if load_dataset is not None:
        try:
            ds = load_dataset(dataset_name)
            ds_path = dataset_dir / "dataset"
            ds.save_to_disk(str(ds_path))
            return ds
        except Exception:
            pass

    return _load_ncc_as_dataframe(dataset_dir, max_rows=max_rows)
