# CLAUDE.md - Project Instructions

This project processes kernel conversion datasets for SFT training on Triton kernel code.

## Project Structure

```
dp-data-pipeline/
├── scripts/                              # Python scripts
│   ├── filter_and_enrich_by_difficulty.py   # Filter & evaluate difficulty
│   ├── generate_prompts.py                  # Synthetic prompt generation
│   ├── filter_unique_best.py                # Deduplication by module
│   ├── analyze_seq_length.py                # Sequence length analysis
│   ├── remove_reasoning.py                  # Remove reasoning from completions
│   └── upload_datasets.py                   # Upload to HuggingFace with splits
├── outputs/           # Generated datasets (gitignored)
│   ├── filtered_dataset.jsonl
│   ├── filtered_dataset-filtered.jsonl
│   └── synthetic_prompts.jsonl
├── .env               # API credentials (gitignored)
├── pyproject.toml     # Project dependencies
└── CLAUDE.md          # This file
```

## Quick Start

```bash
# Install dependencies
uv sync

# Run complete pipeline
uv run python scripts/filter_and_enrich_by_difficulty.py
uv run python scripts/generate_prompts.py
uv run python scripts/filter_unique_best.py
uv run python scripts/analyze_seq_length.py
uv run python scripts/upload_datasets.py all

# Optional: Remove reasoning from uploaded datasets
# uv run python scripts/remove_reasoning.py filtered
# uv run python scripts/remove_reasoning.py unique
```

## Environment Variables

Required in `.env`:
```
PRIME_API_KEY=your_prime_intellect_api_key
PRIME_TEAM_ID=your_team_id
HF_TOKEN=your_huggingface_token
DATASET_PATH=siro1/kernelbook-glm4_7-evals-filtered
FILTERED_DATASET_PATH=outputs/filtered_dataset.jsonl
UNIQUE_DATASET_PATH=outputs/filtered_dataset-filtered.jsonl
SYNTHETIC_DATASET_PATH=outputs/synthetic_prompts.jsonl
```

## Workflow

### 0. Generate Base Dataset (Optional)
Generates the initial `siro1/kernelbook-glm4_7-evals` dataset using prime-rl synthesis pipeline.

**Prerequisites:**
- Install prime-rl from https://github.com/S1ro1/prime-rl
- Start vLLM inference server:
```bash
vllm serve --tensor-parallel-size=8 --async-scheduling --stream-interval 8 \
  --enable-chunked-prefill --speculative-config.method mtp \
  --speculative-config.num-speculative-tokens 1 --enable-prefix-caching \
  zai-org/GLM-4.7-FP8
```

**Run synthesis:**
```bash
uv run synthesize @ configs/synth.toml
```

**Output:** Evaluation in `outputs/` directory (manually upload to HuggingFace as `siro1/kernelbook-glm4_7-evals`)

### 1. Filter and Enrich by Difficulty
Evaluates samples from `siro1/kernelbook-glm4_7-evals` with reward > 0.85 using GPT-5.2.
Adds difficulty rating (low/medium/high) to each sample.

```bash
# Full run (~7,181 samples)
uv run python scripts/filter_and_enrich_by_difficulty.py

# Test mode (5 samples)
TEST_MODE=true uv run python scripts/filter_and_enrich_by_difficulty.py
```

Output: `outputs/filtered_dataset.jsonl` with all original columns + `difficulty`, `evaluation_raw`

### 2. Generate Synthetic Prompts
Generates task specifications from PyTorch modules using GPT-5.2. Creates (prompt, module_name, python_code) tuples for SFT training.

```bash
# Full run (18,162 samples)
uv run python scripts/generate_prompts.py

# Test mode (5 samples)
TEST_MODE=true uv run python scripts/generate_prompts.py
```

Output: `outputs/synthetic_prompts.jsonl` with fields:
- `prompt`: Generated task specification
- `module_name`: Original module name
- `python_code`: Original PyTorch implementation

### 3. Deduplicate by Module
Keeps only the best sample (highest reward) per module name.

```bash
uv run python scripts/filter_unique_best.py
```

Output: `outputs/filtered_dataset-filtered.jsonl` (~2,967 unique modules)

### 4. Analyze Sequence Lengths
Analyzes token sequence lengths to determine optimal `seq_len` for SFT training.

```bash
uv run python scripts/analyze_seq_length.py
```

Provides statistics and recommendations for training configuration.

### 5. Upload to HuggingFace
Uploads datasets with 90/10 train/validation splits.

```bash
# Upload all datasets
uv run python scripts/upload_datasets.py all

# Upload individual datasets
uv run python scripts/upload_datasets.py filtered
uv run python scripts/upload_datasets.py unique
uv run python scripts/upload_datasets.py synthetic
```

### 6. Remove Reasoning (Optional)
Removes reasoning from completions, keeping only the answer field. Requires datasets on HuggingFace first.

```bash
# Process any dataset (raw/filtered/unique)
uv run python scripts/remove_reasoning.py raw
uv run python scripts/remove_reasoning.py filtered
uv run python scripts/remove_reasoning.py unique
```

Parses `<answer>...</answer>` from completion content, removes all reasoning, and uploads with `-no-reasoning` suffix.

## Evaluation Criteria

### Difficulty (low/medium/high)
- **low**: Trivial copy/identity kernel, simple elementwise operations (relu, add)
- **medium**: Basic reductions, standard matrix operations
- **high**: Fused operations, complex attention mechanisms, full model architectures

## HuggingFace Datasets

- **Source:** https://huggingface.co/datasets/GPUMODE/KernelBook (18,162 samples) - Original PyTorch modules
- **Base:** https://huggingface.co/datasets/siro1/kernelbook-glm4_7-evals (18,162 samples) - Generated using prime-rl synthesis with GLM-4.7-FP8
- **Filtered:** https://huggingface.co/datasets/siro1/kernelbook-glm4_7-evals-filtered (~7,181 samples, 90/10 split) - Reward > 0.85 with difficulty ratings
- **Unique:** https://huggingface.co/datasets/siro1/kernelbook-glm4_7-evals-unique (~2,967 samples, 90/10 split) - Deduplicated by module
- **Synthetic:** https://huggingface.co/datasets/siro1/kernelbook-synthetic-tasks (18,162 samples, no split) - Task specifications
- **Filtered (No Reasoning):** https://huggingface.co/datasets/siro1/kernelbook-glm4_7-evals-filtered-no-reasoning (~7,181 samples, 90/10 split) - Filtered with reasoning removed
- **Unique (No Reasoning):** https://huggingface.co/datasets/siro1/kernelbook-glm4_7-evals-unique-no-reasoning (~2,967 samples, 90/10 split) - Unique with reasoning removed

## Notes

- All scripts should be run from the project root directory
- Large dataset files are gitignored - regenerate or download from HuggingFace
- The Prime Intellect API uses OpenAI-compatible endpoints at `https://api.pinference.ai/api/v1`
- Base dataset generation (step 0) requires prime-rl from https://github.com/S1ro1/prime-rl and vLLM
- Step 0 is optional - you can start from the existing `siro1/kernelbook-glm4_7-evals` dataset on HuggingFace

## Code Rules
- Do not write code that can silently produce wrong results, instead code should explicitly fail
- Never do hacks, that can silently cause wrong results, i.e. truncating code inside of prompts, etc.
- Write code that is easy to understand, if section of code is not straightforward, add comments
