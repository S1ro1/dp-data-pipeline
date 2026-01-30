#!/usr/bin/env python3
"""
Synthetic data generation pipeline for Triton kernel task specifications.
Uses GPT-5.2 to generate detailed task prompts from (module_name, python_code).
Output: (prompt, module_name, python_code) tuples.
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
MODEL = "moonshotai/kimi-k2-0905"
BASE_URL = "https://api.pinference.ai/api/v1"
BATCH_SIZE = 1024
OUTPUT_FILE = "outputs/synthetic_prompts.jsonl"
MAX_RETRIES = 3
TEST_MODE = os.environ.get("TEST_MODE", "").lower() == "true"
TEST_SAMPLES = 10

# Source: original unfiltered dataset
SOURCE_DATASET = "GPUMODE/KernelBook"

PROMPT_GENERATION_TEMPLATE = """You are an expert technical writer. Given a PyTorch module, write a clear task specification for implementing it in Triton.

Your specification should include:

1. **Task Description**: What the module computes (mathematical/functional description)
2. **Input Specification**: Tensor shapes, dtypes, and what each input represents
3. **Output Specification**: Expected output shape, dtype, and semantics
4. **Module Interface**:
   - The new module class should be named `{module_name}New`
   - Specify the expected `__init__` parameters
   - Specify the `forward` method signature
5. **Behavioral Requirements**: Edge cases, any constraints
6. **Implementation Language**: Triton (OpenAI's GPU programming language)
7. **Learnable Parameters**: If explicitly specified in the reference, specify the learnable parameters, if reference computes only forward pass, do not specify any learnable parameters

Do NOT include:
- Specific Triton implementation details (block sizes, memory patterns, etc.)
- Code snippets or pseudocode from the original PyTorch implementation
- References to how Triton works internally
- Assumptions about the implementation - the reference is a source of truth, do not make assumptions about the implementation, i.e. reduction dtype, etc.

The specification should be detailed enough that a developer familiar with Triton could implement a correct, functionally equivalent module. Your specification CAN NOT reference the original PyTorch implementation or any of its functions, classes or specifics of the test cases.

---

**Module Name:** {module_name}

**PyTorch Implementation:**
```python
{python_code}
```

---

Write the task specification:

<specification>
[Your detailed specification here]
</specification>"""


OUTPUT_FORMAT = """
Output ONLY valid Python code between the markers below.

BEGIN_PYTHON
END_PYTHON
"""


def extract_specification(text: str) -> str | None:
    """Extract specification from XML-style tags."""
    pattern = r"<specification>\s*(.*?)\s*</specification>"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return None


async def generate_prompt(
    client: AsyncOpenAI,
    sample: dict,
    idx: int,
    semaphore: asyncio.Semaphore,
    pbar: tqdm_asyncio,
) -> dict | None:
    """Generate a task specification for a single sample."""
    async with semaphore:
        module_name = sample["module_name"]
        python_code = sample["python_code"]
        triton_code = sample["triton_code"]
        uuid = sample["uuid"]

        generation_prompt = PROMPT_GENERATION_TEMPLATE.format(
            module_name=module_name,
            python_code=python_code,
        )

        for attempt in range(MAX_RETRIES):
            try:
                response = await client.chat.completions.create(
                    model=MODEL,
                    messages=[{"role": "user", "content": generation_prompt}],
                    max_tokens=32768,
                    temperature=0.3,
                )

                raw_response = response.choices[0].message.content
                specification = extract_specification(raw_response)

                if not specification:
                    if attempt < MAX_RETRIES - 1:
                        continue
                    pbar.update(1)
                    return None

                specification = f"{specification}\n{OUTPUT_FORMAT}"

                pbar.update(1)

                return {
                    "prompt": specification,
                    "module_name": module_name,
                    "python_code": python_code,
                    "triton_code": triton_code,
                    "uuid": uuid,
                }

            except Exception as e:
                if attempt < MAX_RETRIES - 1:
                    await asyncio.sleep(2**attempt)
                else:
                    print(f"\nFailed sample {idx}: {e}")
                    pbar.update(1)
                    return None


async def main():
    print("=" * 60)
    print("Synthetic Prompt Generation Pipeline")
    print("=" * 60)

    print(f"\nLoading dataset: {SOURCE_DATASET}...")
    ds = load_dataset(SOURCE_DATASET, split="train")

    samples = list(ds)
    if TEST_MODE:
        print(f"\n*** TEST MODE: {TEST_SAMPLES} samples ***")
        samples = samples[:TEST_SAMPLES]

    # Print metrics
    print("\n" + "-" * 40)
    print("DATASET METRICS")
    print("-" * 40)
    print(f"Total samples to process:  {len(samples):,}")
    print(f"Model: {MODEL}")
    print(f"Batch size (concurrent): {BATCH_SIZE}")
    print(f"Output file: {OUTPUT_FILE}")
    print("-" * 40)

    client = AsyncOpenAI(
        api_key=os.environ["PRIME_API_KEY"],
        base_url=BASE_URL,
        default_headers={"X-Prime-Team-ID": os.environ["PRIME_TEAM_ID"]},
    )

    semaphore = asyncio.Semaphore(BATCH_SIZE)
    start_time = time.time()

    with tqdm_asyncio(total=len(samples), desc="Generating") as pbar:
        tasks = [
            generate_prompt(client, s, i, semaphore, pbar)
            for i, s in enumerate(samples)
        ]
        results = await asyncio.gather(*tasks)

    elapsed = time.time() - start_time

    successful = [r for r in results if r is not None]
    failed_count = len(results) - len(successful)

    # Write results
    output_path = Path(OUTPUT_FILE)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        for result in successful:
            f.write(json.dumps(result) + "\n")

    # Final statistics
    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)
    print(f"Total processed:        {len(results):,}")
    print(f"Successful:             {len(successful):,}")
    print(f"Failed/Skipped:         {failed_count:,}")
    print(f"Time elapsed:           {elapsed:.1f}s")
    print(f"Rate:                   {len(results) / elapsed:.1f} samples/sec")
    print(f"Output saved to:        {OUTPUT_FILE}")

    # Prompt length statistics
    if successful:
        prompt_lengths = [len(r["prompt"]) for r in successful]
        print(f"\nGenerated prompt lengths (chars):")
        print(f"  Min: {min(prompt_lengths):,}")
        print(f"  Max: {max(prompt_lengths):,}")
        print(f"  Avg: {sum(prompt_lengths) // len(prompt_lengths):,}")

    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
