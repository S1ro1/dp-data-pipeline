#!/usr/bin/env python3
"""
Batched filtering script for kernelbook-glm4-evals dataset.
Evaluates samples using GPT-5.2 to determine difficulty and style ratings.
Keeps ALL original columns for SFT training.
"""

import asyncio
import json
import os
import re
import time
from pathlib import Path

from datasets import load_dataset
from dotenv import load_dotenv
from openai import AsyncOpenAI
from tqdm.asyncio import tqdm_asyncio

load_dotenv()

# Configuration
MODEL = "openai/gpt-5.2"
BASE_URL = "https://api.pinference.ai/api/v1"
REWARD_THRESHOLD = 0.85
BATCH_SIZE = 256  # Concurrent requests
OUTPUT_FILE = "outputs/filtered_dataset.jsonl"
MAX_RETRIES = 3
TEST_MODE = os.environ.get("TEST_MODE", "").lower() == "true"
TEST_SAMPLES = 5  # Number of samples for test mode


EVALUATION_PROMPT = """You are an expert evaluator for CUDA/Triton kernel code. Given a kernel conversion task and its solution, provide a difficulty rating.

## Difficulty Rating (low, medium, high): How hard is this kernel conversion task?

- low: Trivial copy/identity kernel, elementwise operations
- medium: basic reductions, matrix operations
- high: fused operations, full model architectures

---

**Task:** Convert this PyTorch module to Triton kernels.

**Module:** {module_name}

**Original PyTorch:**
```python
{python_code}
```

**Solution:**
{completion}

---

Respond ONLY with:
<difficulty>{difficulty}</difficulty>
"""


def extract_difficulty(text: str) -> str | None:
    """Extract difficulty rating from text."""
    pattern = r"<difficulty>(low|medium|high)</difficulty>"
    match = re.search(pattern, text)
    if match:
        return match.group(1).lower()
    return None


async def evaluate_sample(
    client: AsyncOpenAI,
    sample: dict,
    idx: int,
    semaphore: asyncio.Semaphore,
    pbar: tqdm_asyncio,
) -> dict | None:
    """Evaluate a single sample using the inference API. Returns full sample with added fields."""
    async with semaphore:
        info = sample.get("info", {}) or {}
        module_name = info.get("module_name", "Unknown")
        python_code = info.get("python_code", "")

        # Extract completion text
        completion_messages = sample.get("completion", [])
        if completion_messages and isinstance(completion_messages, list):
            completion = (
                completion_messages[0].get("content", "") if completion_messages else ""
            )
        else:
            completion = str(completion_messages)

        prompt = EVALUATION_PROMPT.format(
            module_name=module_name,
            python_code=python_code,
            completion=completion,
        )

        for attempt in range(MAX_RETRIES):
            try:
                response = await client.chat.completions.create(
                    model=MODEL,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=256,
                    temperature=0.1,
                )

                evaluation = response.choices[0].message.content
                difficulty = extract_difficulty(evaluation)

                pbar.update(1)

                # Return full sample dict with added evaluation fields
                result = dict(sample)  # Copy all original fields
                result["difficulty"] = difficulty
                result["evaluation_raw"] = evaluation

                return result

            except Exception as e:
                if attempt < MAX_RETRIES - 1:
                    await asyncio.sleep(2**attempt)  # Exponential backoff
                else:
                    print(
                        f"\nFailed to evaluate sample {sample.get('example_id', idx)}: {e}"
                    )
                    pbar.update(1)
                    return None


async def main():
    print("=" * 60)
    print("Dataset Filtering Script (Full Columns for SFT)")
    print("=" * 60)

    # Load dataset
    print("\nLoading dataset...")
    ds = load_dataset("siro1/kernelbook-glm4_7-evals", split="train")
    total_samples = len(ds)

    print(f"Original columns: {ds.column_names}")

    # Filter by reward threshold
    filtered_indices = [
        i for i, sample in enumerate(ds) if sample["reward"] > REWARD_THRESHOLD
    ]
    samples_to_evaluate = [ds[i] for i in filtered_indices]

    # Test mode: limit samples
    if TEST_MODE:
        print(f"\n*** TEST MODE: Limiting to {TEST_SAMPLES} samples ***")
        samples_to_evaluate = samples_to_evaluate[:TEST_SAMPLES]

    # Print metrics
    print("\n" + "-" * 40)
    print("DATASET METRICS")
    print("-" * 40)
    print(f"Total samples in dataset:     {total_samples:,}")
    print(f"Reward threshold:             > {REWARD_THRESHOLD}")
    print(f"Samples passing threshold:    {len(samples_to_evaluate):,}")
    print(f"Samples skipped:              {total_samples - len(samples_to_evaluate):,}")
    print(
        f"Percentage to evaluate:       {len(samples_to_evaluate) / total_samples * 100:.1f}%"
    )
    print("-" * 40)

    rewards = [s["reward"] for s in samples_to_evaluate]
    print("\nFiltered samples reward stats:")
    print(f"  Min reward:  {min(rewards):.4f}")
    print(f"  Max reward:  {max(rewards):.4f}")
    print(f"  Avg reward:  {sum(rewards) / len(rewards):.4f}")

    print(f"\nModel: {MODEL}")
    print(f"Batch size (concurrent): {BATCH_SIZE}")
    print(f"Output file: {OUTPUT_FILE}")
    print("Columns: ALL original + difficulty, style, evaluation_raw")
    print("=" * 60)

    # Initialize client
    client = AsyncOpenAI(
        api_key=os.environ["PRIME_API_KEY"],
        base_url=BASE_URL,
        default_headers={"X-Prime-Team-ID": os.environ["PRIME_TEAM_ID"]},
    )

    semaphore = asyncio.Semaphore(BATCH_SIZE)

    print(f"\nStarting evaluation of {len(samples_to_evaluate):,} samples...")
    start_time = time.time()

    with tqdm_asyncio(total=len(samples_to_evaluate), desc="Evaluating") as pbar:
        tasks = [
            evaluate_sample(client, sample, idx, semaphore, pbar)
            for idx, sample in enumerate(samples_to_evaluate)
        ]
        results = await asyncio.gather(*tasks)

    elapsed = time.time() - start_time

    successful_results = [r for r in results if r is not None]
    failed_count = len(results) - len(successful_results)

    output_path = Path(OUTPUT_FILE)
    with output_path.open("w") as f:
        for result in successful_results:
            f.write(json.dumps(result) + "\n")

    # Final statistics
    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)
    print(f"Total evaluated:        {len(results):,}")
    print(f"Successful:             {len(successful_results):,}")
    print(f"Failed:                 {failed_count:,}")
    print(f"Time elapsed:           {elapsed:.1f}s")
    print(f"Rate:                   {len(results) / elapsed:.1f} samples/sec")
    print(f"Output saved to:        {OUTPUT_FILE}")

    difficulties = [
        r["difficulty"] for r in successful_results if r.get("difficulty") is not None
    ]

    for diff in ("low", "medium", "high"):
        count = difficulties.count(diff)
        print(f"  {diff}: {count:4d} ({count / len(difficulties) * 100:.1f}%)")

    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
