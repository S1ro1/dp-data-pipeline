#!/usr/bin/env python3
"""
Upload filtered and unique datasets to HuggingFace with train/validation splits.
"""

import json
import os
from pathlib import Path

from datasets import Dataset, DatasetDict
from dotenv import load_dotenv
from huggingface_hub import login

load_dotenv()

# Configuration
FILTERED_FILE = "outputs/filtered_dataset.jsonl"
UNIQUE_FILE = "outputs/filtered_dataset-filtered.jsonl"
FILTERED_REPO = "siro1/kernelbook-glm4-evals-filtered"
UNIQUE_REPO = "siro1/kernelbook-glm4-evals-unique"
SPLIT_RATIO = 0.1  # 10% validation
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


def main():
    # Login to HuggingFace
    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        raise ValueError("HF_TOKEN not found in environment")
    login(token=hf_token)
    print("Logged in to HuggingFace")

    # Upload filtered dataset (all samples with reward > 0.85 and evaluations)
    filtered_data = load_jsonl(FILTERED_FILE)
    upload_with_splits(
        filtered_data,
        FILTERED_REPO,
        "Filtered kernelbook samples with difficulty/style evaluations"
    )

    # Upload unique dataset (deduplicated, best per module)
    unique_data = load_jsonl(UNIQUE_FILE)
    upload_with_splits(
        unique_data,
        UNIQUE_REPO,
        "Unique kernelbook samples (best per module)"
    )

    print(f"\n{'=' * 60}")
    print("UPLOAD COMPLETE")
    print(f"{'=' * 60}")
    print(f"Filtered: https://huggingface.co/datasets/{FILTERED_REPO}")
    print(f"Unique: https://huggingface.co/datasets/{UNIQUE_REPO}")


if __name__ == "__main__":
    main()
