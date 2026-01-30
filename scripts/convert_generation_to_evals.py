#!/usr/bin/env python3
"""
Convert generation output to base evals format.

Takes generation output (with trajectory data) and converts it to the format
expected by downstream scripts (matching siro1/kernelbook-{model}-evals).

Usage:
    uv run python scripts/convert_generation_to_evals.py generation_output.jsonl
    uv run python scripts/convert_generation_to_evals.py generation_output.jsonl --no-upload
    TEST_MODE=true uv run python scripts/convert_generation_to_evals.py generation_output.jsonl

Input format (generation output):
    - example_id, prompt, completion, answer, task, info, reward, error,
      total_ms, generation_ms, scoring_ms, speedup_reward, num_turns, trajectory
    - trajectory contains ChatCompletion with reasoning_content

Output format (base evals):
    - example_id, prompt, completion (with reasoning field), task, reward,
      generation_ms, scoring_ms, total_ms, info, answer, speedup_reward,
      num_turns, oai_tools
"""

import argparse
import os
from pathlib import Path

import config

TEST_MODE = os.environ.get("TEST_MODE", "").lower() == "true"
TEST_SAMPLES = 10


def extract_reasoning_from_trajectory(trajectory: list[dict]) -> str:
    last_turn = trajectory[-1]
    response = last_turn["response"]

    return response["choices"][0]["message"]["reasoning_content"]


def convert_sample(sample: dict) -> dict:
    """
    Convert a single sample from generation format to base evals format.

    Ensures the output matches the exact schema of siro1/kernelbook-{model}-evals.
    Raises KeyError if required fields are missing.
    """
    # Extract reasoning from trajectory
    reasoning = extract_reasoning_from_trajectory(sample["trajectory"])

    # Build completion with reasoning field
    # Original completion structure: [{"role": "assistant", "content": "..."}]
    # Target structure: [{"role": "assistant", "content": "...", "reasoning": "..."}]
    original_completion = sample["completion"]

    new_completion = []
    for msg in original_completion:
        new_msg = {
            "role": msg["role"],
            "content": msg["content"],
            "reasoning": reasoning,
        }
        new_completion.append(new_msg)

    # Build the output record with exact column order matching target format
    result = {
        "example_id": sample["example_id"],
        "prompt": sample["prompt"],
        "completion": new_completion,
        "task": sample["task"],
        "reward": sample["reward"],
        "generation_ms": sample["generation_ms"],
        "scoring_ms": sample["scoring_ms"],
        "total_ms": sample["total_ms"],
        "info": sample["info"],
        "answer": sample["answer"],
        "speedup_reward": sample["speedup_reward"],
        "num_turns": sample["num_turns"],
        "oai_tools": None,  # Not present in generation output, set to null
    }

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Convert generation output to base evals format"
    )
    parser.add_argument(
        "input_file",
        type=Path,
        help="Path to input JSONL file (generation output)",
    )
    parser.add_argument(
        "--no-upload",
        action="store_true",
        help="Skip uploading to HuggingFace",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=None,
        help="Output path (default: outputs/{model}/evals_dataset.jsonl)",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("Convert Generation Output to Evals Format")
    print("=" * 60)
    config.print_config()

    # Determine output path
    output_path = args.output or config.get_output_dir() / "evals_dataset.jsonl"

    # Load input data
    input_path = args.input_file
    print(f"\nLoading input file: {input_path}")

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    data = config.load_jsonl(input_path)
    print(f"Loaded {len(data):,} samples")

    # Test mode: limit samples
    if TEST_MODE:
        print(f"\n*** TEST MODE: Limiting to {TEST_SAMPLES} samples ***")
        data = data[:TEST_SAMPLES]

    # Convert samples
    print(f"\nConverting {len(data):,} samples...")
    converted = []

    for idx, sample in enumerate(data):
        try:
            result = convert_sample(sample)
            converted.append(result)
        except KeyError as e:
            raise KeyError(
                f"Sample {idx} (example_id={sample.get('example_id', 'unknown')}) "
                f"missing required field: {e}"
            ) from e

    # Save output
    config.save_jsonl(converted, output_path)

    # Print statistics
    print("\n" + "=" * 60)
    print("CONVERSION RESULTS")
    print("=" * 60)
    print(f"Input samples:              {len(data):,}")
    print(f"Output samples:             {len(converted):,}")

    rewards = [s["reward"] for s in converted]
    print("\nReward statistics:")
    print(f"  Min:  {min(rewards):.4f}")
    print(f"  Max:  {max(rewards):.4f}")
    print(f"  Avg:  {sum(rewards) / len(rewards):.4f}")

    print(f"\nOutput columns: {list(converted[0].keys())}")

    # Upload to HuggingFace
    if not args.no_upload:
        config.hf_login()
        # Upload without splits for base evals dataset
        config.upload_without_splits(converted, config.get_source_repo())

    print("=" * 60)


if __name__ == "__main__":
    main()
