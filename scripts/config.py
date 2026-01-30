#!/usr/bin/env python3
"""
Shared configuration for dp-data-pipeline scripts.

All paths and repo names are derived from DATA_MODEL environment variable.
"""

import json
import os
from pathlib import Path

from datasets import Dataset, DatasetDict
from dotenv import load_dotenv
from huggingface_hub import login

load_dotenv()

# Model name from environment (e.g., "glm4_7", "qwen3_30b", etc.)
DATA_MODEL = os.environ.get("DATA_MODEL", "glm4_7")

# HuggingFace organization
HF_ORG = "siro1"

# Split configuration
SPLIT_RATIO = 0.1
SEED = 42


def get_output_dir() -> Path:
    """Get output directory for current model: outputs/{model}/"""
    path = Path("outputs") / DATA_MODEL
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_filtered_path() -> Path:
    """Get path to filtered dataset JSONL file."""
    return get_output_dir() / "filtered_dataset.jsonl"


def get_unique_path() -> Path:
    """Get path to unique (deduplicated) dataset JSONL file."""
    return get_output_dir() / "unique_dataset.jsonl"


def get_synthetic_path() -> Path:
    """Get path to synthetic prompts JSONL file."""
    return get_output_dir() / "synthetic_prompts.jsonl"


def get_source_repo() -> str:
    """Get HuggingFace repo for source evals dataset."""
    return f"{HF_ORG}/kernelbook-{DATA_MODEL}-evals"


def get_filtered_repo() -> str:
    """Get HuggingFace repo for filtered dataset."""
    return f"{HF_ORG}/kernelbook-{DATA_MODEL}-evals-filtered"


def get_unique_repo() -> str:
    """Get HuggingFace repo for unique dataset."""
    return f"{HF_ORG}/kernelbook-{DATA_MODEL}-evals-unique"


def get_synthetic_repo() -> str:
    """Get HuggingFace repo for synthetic prompts dataset."""
    return f"{HF_ORG}/kernelbook-{DATA_MODEL}-synthetic-tasks"


def get_no_reasoning_repo(base_repo: str) -> str:
    """Get HuggingFace repo name with -no-reasoning suffix."""
    return f"{base_repo}-no-reasoning"


def load_jsonl(filepath: Path) -> list[dict]:
    """Load data from JSONL file."""
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")
    data = []
    with filepath.open("r") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


def save_jsonl(data: list[dict], filepath: Path) -> None:
    """Save data to JSONL file."""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with filepath.open("w") as f:
        for item in data:
            f.write(json.dumps(item) + "\n")
    print(f"Saved {len(data):,} samples to: {filepath}")


def hf_login() -> None:
    """Login to HuggingFace using HF_TOKEN from environment."""
    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        raise ValueError("HF_TOKEN not found in environment")
    login(token=hf_token)
    print("Logged in to HuggingFace")


def upload_with_splits(data: list[dict], repo_name: str) -> str:
    """
    Upload dataset with train/validation splits (90/10).
    Returns the HuggingFace URL.
    """
    print(f"\n{'=' * 60}")
    print(f"Uploading: {repo_name}")
    print(f"{'=' * 60}")
    print(f"Total samples: {len(data):,}")

    ds = Dataset.from_list(data)
    split = ds.train_test_split(test_size=SPLIT_RATIO, seed=SEED)

    dataset_dict = DatasetDict({"train": split["train"], "validation": split["test"]})

    print(f"Train samples: {len(dataset_dict['train']):,}")
    print(f"Validation samples: {len(dataset_dict['validation']):,}")

    dataset_dict.push_to_hub(repo_name)
    url = f"https://huggingface.co/datasets/{repo_name}"
    print(f"Uploaded to: {url}")
    return url


def upload_without_splits(data: list[dict], repo_name: str) -> str:
    """
    Upload dataset without splits.
    Returns the HuggingFace URL.
    """
    print(f"\n{'=' * 60}")
    print(f"Uploading: {repo_name}")
    print(f"{'=' * 60}")
    print(f"Total samples: {len(data):,}")

    ds = Dataset.from_list(data)
    ds.push_to_hub(repo_name)
    url = f"https://huggingface.co/datasets/{repo_name}"
    print(f"Uploaded to: {url}")
    return url


def print_config():
    """Print current configuration."""
    print(f"DATA_MODEL: {DATA_MODEL}")
    print(f"Output directory: {get_output_dir()}")
    print(f"Source repo: {get_source_repo()}")
    print(f"Filtered repo: {get_filtered_repo()}")
    print(f"Unique repo: {get_unique_repo()}")
    print(f"Synthetic repo: {get_synthetic_repo()}")
