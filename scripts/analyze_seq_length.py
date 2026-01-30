#!/usr/bin/env python3
"""
Analyze sequence lengths in a dataset to determine optimal seq_len for SFT.

Usage:
    uv run python scripts/analyze_seq_length.py              # Analyze filtered dataset
    uv run python scripts/analyze_seq_length.py --unique     # Analyze unique dataset
    uv run python scripts/analyze_seq_length.py --source     # Analyze source evals dataset
"""

import argparse

from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer

import config

TOKENIZER_NAME = "Qwen/Qwen3-30B-A3B-Thinking-2507"


def main():
    parser = argparse.ArgumentParser(description="Analyze sequence lengths")
    parser.add_argument(
        "--unique",
        action="store_true",
        help="Analyze unique dataset instead of filtered",
    )
    parser.add_argument(
        "--source",
        action="store_true",
        help="Analyze source evals dataset",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("Sequence Length Analysis")
    print("=" * 60)
    config.print_config()

    # Determine which dataset to analyze
    if args.source:
        dataset_path = config.get_source_repo()
    elif args.unique:
        dataset_path = config.get_unique_repo()
    else:
        dataset_path = config.get_filtered_repo()

    print(f"\nLoading tokenizer: {TOKENIZER_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME, trust_remote_code=True)

    print(f"Loading dataset: {dataset_path}")
    data = load_dataset(dataset_path, split="train")
    print(f"Loaded {len(data)} samples")

    # Tokenize each sample using chat template
    lengths = []
    for sample in tqdm(data, desc="Tokenizing"):
        # The dataset has 'prompt' (list of messages) and 'completion' (list with assistant response)
        prompt_messages = sample.get("prompt", [])
        completion_messages = sample.get("completion", [])

        # Combine prompt and completion into full conversation
        messages = prompt_messages + completion_messages

        # Apply chat template
        try:
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )
            tokens = tokenizer.encode(text, add_special_tokens=False)
            lengths.append(len(tokens))
        except Exception as e:
            print(f"Error tokenizing sample: {e}")
            continue

    lengths_sorted = sorted(lengths)

    print("\n" + "=" * 60)
    print("SEQUENCE LENGTH STATISTICS")
    print("=" * 60)
    print(f"Total samples: {len(lengths)}")
    print(f"Min length: {min(lengths)}")
    print(f"Max length: {max(lengths)}")
    print(f"Mean length: {sum(lengths) / len(lengths):.1f}")
    print(f"Median length: {lengths_sorted[len(lengths) // 2]}")

    # Percentiles
    percentiles = [50, 75, 90, 95, 99, 99.5, 99.9, 100]
    print("\nPercentiles:")
    for p in percentiles:
        idx = int(len(lengths_sorted) * p / 100) - 1
        idx = max(0, min(idx, len(lengths_sorted) - 1))
        print(f"  {p:5.1f}%: {lengths_sorted[idx]:,} tokens")

    # Distribution buckets
    print("\nLength distribution:")
    buckets = [1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072]
    for i, bucket in enumerate(buckets):
        count = sum(1 for length in lengths if length <= bucket)
        prev_bucket = buckets[i - 1] if i > 0 else 0
        in_bucket = sum(1 for length in lengths if prev_bucket < length <= bucket)
        pct = count / len(lengths) * 100
        print(f"  <= {bucket:6,}: {count:5,} ({pct:5.1f}%) | in bucket: {in_bucket:,}")

    # Count samples over common limits
    print("\nSamples exceeding common limits:")
    limits = [2048, 4096, 8192, 16384, 32768]
    for limit in limits:
        over = sum(1 for length in lengths if length > limit)
        pct = over / len(lengths) * 100
        print(f"  > {limit:5,}: {over:,} ({pct:.1f}%)")

    # Recommend seq_len
    print("\n" + "=" * 60)
    print("RECOMMENDATIONS")
    print("=" * 60)
    p99 = lengths_sorted[int(len(lengths_sorted) * 0.99) - 1]
    p995 = lengths_sorted[int(len(lengths_sorted) * 0.995) - 1]

    # Round up to nearest power of 2 or common value
    def round_to_common(n):
        common = [1024, 2048, 4096, 8192, 16384, 32768, 65536]
        for c in common:
            if n <= c:
                return c
        return common[-1]

    rec_99 = round_to_common(p99)
    rec_995 = round_to_common(p995)

    print(f"For 99% coverage: seq_len = {rec_99:,}")
    print(f"For 99.5% coverage: seq_len = {rec_995:,}")
    print(f"For 100% coverage: seq_len = {round_to_common(max(lengths)):,}")


if __name__ == "__main__":
    main()
