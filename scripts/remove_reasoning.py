#!/usr/bin/env python3
"""
Remove reasoning from completion content, keeping only the answer.
Parses <answer>...</answer> from completion and removes all reasoning.

Note: Datasets must be uploaded to HuggingFace first (run upload_datasets.py).

Usage:
    uv run python scripts/remove_reasoning.py raw        # Process base dataset
    uv run python scripts/remove_reasoning.py filtered   # Process filtered dataset
    uv run python scripts/remove_reasoning.py unique     # Process unique dataset
"""

import argparse
import os
import re

from datasets import Dataset, DatasetDict, load_dataset
from dotenv import load_dotenv
from huggingface_hub import login
from tqdm import tqdm

load_dotenv()

# Dataset configuration
DATASETS = {
    "raw": "siro1/kernelbook-glm4_7-evals",
    "filtered": "siro1/kernelbook-glm4_7-evals-filtered",
    "unique": "siro1/kernelbook-glm4_7-evals-unique",
}

SPLIT_RATIO = 0.1
SEED = 42


def extract_answer(content: str) -> str:
    """
    Extract answer from <answer>...</answer> tags in completion content.

    Raises ValueError if no answer tags found.
    """
    pattern = r"<answer>(.*?)</answer>"
    match = re.search(pattern, content, re.DOTALL)
    if not match:
        raise ValueError("No <answer> tags found in completion content")
    return match.group(1)


def process_sample(sample: dict) -> dict:
    """
    Remove reasoning from completion, keeping only the answer.

    Parses answer from completion[0]["content"] and replaces entire content
    with just <answer>{parsed_answer}</answer>.
    """
    # Ensure completion exists and has at least one element
    if "completion" not in sample or not sample["completion"]:
        raise ValueError(
            f"Sample missing completion field: {sample['info'].get('module_name', 'unknown')}"
        )

    completion_content = sample["completion"][0].get("content", "")
    if not completion_content:
        raise ValueError(
            f"Sample has empty completion content: {sample['info'].get('module_name', 'unknown')}"
        )

    # Extract answer from completion content
    try:
        answer = extract_answer(completion_content)
    except ValueError as _:
        raise ValueError(
            f"Error processing {sample['info'].get('module_name', 'unknown')}"
        )

    # Create new completion with answer only
    processed = sample.copy()
    processed["completion"] = [
        {
            "role": processed["completion"][0]["role"],
            "content": f"<answer>{answer}</answer>",
        }
    ]

    return processed


def main():
    parser = argparse.ArgumentParser(
        description="Remove reasoning from completion, keeping only answer"
    )
    parser.add_argument(
        "dataset",
        choices=["raw", "filtered", "unique"],
        help="Which dataset to process",
    )
    args = parser.parse_args()

    # Login to HuggingFace
    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        raise ValueError("HF_TOKEN not found in environment")
    login(token=hf_token)
    print("Logged in to HuggingFace")

    # Load dataset
    source_repo = DATASETS[args.dataset]
    target_repo = f"{source_repo}-no-reasoning"

    print(f"\n{'=' * 60}")
    print(f"Processing: {source_repo}")
    print(f"Target: {target_repo}")
    print(f"{'=' * 60}\n")

    # Load dataset (handle both with and without splits)
    try:
        ds = load_dataset(source_repo)
        has_splits = True
        print("Loaded dataset with splits:")
        for split_name, split_data in ds.items():
            print(f"  {split_name}: {len(split_data):,} samples")
    except ValueError:
        # Dataset has no splits, load as single split
        ds = load_dataset(source_repo, split="train")
        has_splits = False
        print(f"Loaded dataset (no splits): {len(ds):,} samples")

    # Process dataset
    if has_splits:
        processed_splits = {}
        for split_name, split_data in ds.items():
            print(f"\nProcessing {split_name} split...")
            processed_data = []
            for sample in tqdm(split_data, desc=f"Processing {split_name}"):
                try:
                    processed_data.append(process_sample(sample))
                except ValueError as e:
                    print(f"\nError: {e}")
                    raise
            processed_splits[split_name] = Dataset.from_list(processed_data)

        dataset_dict = DatasetDict(processed_splits)
    else:
        print("\nProcessing samples...")
        processed_data = []
        for sample in tqdm(ds, desc="Processing"):
            try:
                processed_data.append(process_sample(sample))
            except ValueError as e:
                print(f"\nError: {e}")
                raise
        dataset_dict = Dataset.from_list(processed_data)

    # Upload
    print(f"\n{'=' * 60}")
    print(f"Uploading to: {target_repo}")
    print(f"{'=' * 60}\n")

    dataset_dict.push_to_hub(target_repo)

    print(f"\n{'=' * 60}")
    print("UPLOAD COMPLETE")
    print(f"{'=' * 60}")
    print(f"Uploaded to: https://huggingface.co/datasets/{target_repo}")


if __name__ == "__main__":
    main()
