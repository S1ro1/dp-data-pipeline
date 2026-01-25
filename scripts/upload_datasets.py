#!/usr/bin/env python3
"""
Upload datasets to HuggingFace with train/validation splits.

Usage:
    uv run python scripts/upload_datasets.py              # Upload all datasets
    uv run python scripts/upload_datasets.py synthetic    # Upload only synthetic
    uv run python scripts/upload_datasets.py filtered     # Upload only filtered
    uv run python scripts/upload_datasets.py unique       # Upload only unique
"""

import argparse
import json
import os

from datasets import Dataset, DatasetDict
from dotenv import load_dotenv
from huggingface_hub import login

load_dotenv()

# Configuration
FILTERED_FILE = os.environ["FILTERED_DATASET_PATH"]
UNIQUE_FILE = os.environ["UNIQUE_DATASET_PATH"]
SYNTHETIC_FILE = os.environ["SYNTHETIC_DATASET_PATH"]
FILTERED_REPO = "siro1/kernelbook-glm4_7-evals-filtered"
UNIQUE_REPO = "siro1/kernelbook-glm4_7-evals-unique"
SYNTHETIC_REPO = "siro1/kernelbook-synthetic-tasks"
SPLIT_RATIO = 0.1
SEED = 42


def load_jsonl(filepath: str) -> list[dict]:
    """Load data from JSONL file."""
    data = []
    with open(filepath, "r") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


def upload_with_splits(data: list[dict], repo_name: str, description: str):
    """Upload dataset with train/validation splits."""
    print(f"\n{'=' * 60}")
    print(f"Uploading: {repo_name}")
    print(f"{'=' * 60}")
    print(f"Total samples: {len(data):,}")

    # Create dataset and split
    ds = Dataset.from_list(data)
    split = ds.train_test_split(test_size=SPLIT_RATIO, seed=SEED)

    dataset_dict = DatasetDict({
        "train": split["train"],
        "validation": split["test"]
    })

    print(f"Train samples: {len(dataset_dict['train']):,}")
    print(f"Validation samples: {len(dataset_dict['validation']):,}")

    # Upload
    dataset_dict.push_to_hub(repo_name)
    print(f"Uploaded to: https://huggingface.co/datasets/{repo_name}")


def upload_without_splits(data: list[dict], repo_name: str, description: str):
    """Upload dataset without splits (single train split)."""
    print(f"\n{'=' * 60}")
    print(f"Uploading: {repo_name}")
    print(f"{'=' * 60}")
    print(f"Total samples: {len(data):,}")

    # Create and upload dataset directly
    ds = Dataset.from_list(data)
    ds.push_to_hub(repo_name)
    print(f"Uploaded to: https://huggingface.co/datasets/{repo_name}")


def main():
    parser = argparse.ArgumentParser(description="Upload datasets to HuggingFace")
    parser.add_argument(
        "dataset",
        nargs="?",
        choices=["filtered", "unique", "synthetic", "all"],
        default="all",
        help="Which dataset to upload (default: all)"
    )
    args = parser.parse_args()

    # Login to HuggingFace
    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        raise ValueError("HF_TOKEN not found in environment")
    login(token=hf_token)
    print("Logged in to HuggingFace")

    uploaded = []

    # Upload filtered dataset (all samples with reward > 0.85 and evaluations)
    if args.dataset in ("filtered", "all"):
        filtered_data = load_jsonl(FILTERED_FILE)
        upload_with_splits(
            filtered_data,
            FILTERED_REPO,
            "Filtered kernelbook samples with difficulty/style evaluations"
        )
        uploaded.append(f"Filtered: https://huggingface.co/datasets/{FILTERED_REPO}")

    # Upload unique dataset (deduplicated, best per module)
    if args.dataset in ("unique", "all"):
        unique_data = load_jsonl(UNIQUE_FILE)
        upload_with_splits(
            unique_data,
            UNIQUE_REPO,
            "Unique kernelbook samples (best per module)"
        )
        uploaded.append(f"Unique: https://huggingface.co/datasets/{UNIQUE_REPO}")

    # Upload synthetic prompts dataset (no splits)
    if args.dataset in ("synthetic", "all"):
        synthetic_data = load_jsonl(SYNTHETIC_FILE)
        upload_without_splits(
            synthetic_data,
            SYNTHETIC_REPO,
            "Synthetic task specifications for Triton kernel generation"
        )
        uploaded.append(f"Synthetic: https://huggingface.co/datasets/{SYNTHETIC_REPO}")

    print(f"\n{'=' * 60}")
    print("UPLOAD COMPLETE")
    print(f"{'=' * 60}")
    for url in uploaded:
        print(url)


if __name__ == "__main__":
    main()
