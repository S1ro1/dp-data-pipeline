# dp-data-pipeline

Data processing pipeline for filtering and preparing Triton kernel datasets for SFT (Supervised Fine-Tuning).

## Overview

This pipeline processes PyTorch-to-Triton kernel conversion datasets through filtering, evaluation, and synthetic data generation to create high-quality training datasets for SFT.

**Pipeline Steps:**
0. **(Optional)** Generate base evaluation dataset using prime-rl synthesis
1. Filter high-quality samples (reward > 0.85) and evaluate difficulty
2. Generate synthetic task specifications from PyTorch modules
3. Deduplicate by module name (keep best per module)
4. Analyze sequence lengths for training configuration
5. Upload to HuggingFace with train/val splits
6. **(Optional)** Remove reasoning from completions (requires datasets on HF)

## Datasets

### Source Dataset
- **[GPUMODE/KernelBook](https://huggingface.co/datasets/GPUMODE/KernelBook)** (18,162 samples)
  - Original PyTorch modules with metadata
  - Used as source for synthetic prompt generation

### Generated Base Dataset
- **[siro1/kernelbook-glm4_7-evals](https://huggingface.co/datasets/siro1/kernelbook-glm4_7-evals)** (18,162 samples)
  - Generated using prime-rl synthesis pipeline with GLM-4.7-FP8 model
  - Contains PyTorch modules, generated Triton kernels, and evaluation metrics (reward scores)
  - Created by running vLLM inference + synthesis (step 0 below)

### Processed Datasets
- **[siro1/kernelbook-glm4_7-evals-filtered](https://huggingface.co/datasets/siro1/kernelbook-glm4_7-evals-filtered)** (~7,181 samples, 90/10 split)
  - Filtered subset with reward > 0.85
  - Enriched with GPT-5.2 difficulty ratings: **low**, **medium**, **high**
  - Used for SFT training on high-quality kernel conversions

- **[siro1/kernelbook-glm4_7-evals-unique](https://huggingface.co/datasets/siro1/kernelbook-glm4_7-evals-unique)** (~2,967 samples, 90/10 split)
  - Deduplicated version keeping only best sample per module name
  - Reduces redundancy for more diverse training data

- **[siro1/kernelbook-synthetic-tasks](https://huggingface.co/datasets/siro1/kernelbook-synthetic-tasks)** (18,162 samples)
  - Synthetic task specifications generated from PyTorch modules
  - Each sample: `(prompt, module_name, python_code)` tuple
  - Prompts describe what to implement without implementation details
  - Used for training models to convert natural language specs to Triton

### No-Reasoning Variants (Optional)
- **[siro1/kernelbook-glm4_7-evals-filtered-no-reasoning](https://huggingface.co/datasets/siro1/kernelbook-glm4_7-evals-filtered-no-reasoning)** (~7,181 samples, 90/10 split)
  - Filtered dataset with reasoning removed from completions
  - Contains only `<answer>{answer}</answer>` in completion content

- **[siro1/kernelbook-glm4_7-evals-unique-no-reasoning](https://huggingface.co/datasets/siro1/kernelbook-glm4_7-evals-unique-no-reasoning)** (~2,967 samples, 90/10 split)
  - Unique dataset with reasoning removed from completions
  - Contains only `<answer>{answer}</answer>` in completion content

## Quick Start

```bash
# Install dependencies
uv sync

# Configure .env (see Environment Variables below)

# Run pipeline (see scripts/README.md for details)
uv run python scripts/filter_and_enrich_by_difficulty.py
uv run python scripts/generate_prompts.py
uv run python scripts/filter_unique_best.py
uv run python scripts/analyze_seq_length.py
uv run python scripts/upload_datasets.py all

# Optional: Remove reasoning from uploaded datasets
# uv run python scripts/remove_reasoning.py filtered
# uv run python scripts/remove_reasoning.py unique
```

**Test Mode:** Run with `TEST_MODE=true` to process only 5 samples.

**Step 0 (Optional):** To generate the base dataset from scratch:
```bash
# Requires prime-rl: https://github.com/S1ro1/prime-rl
# Start vLLM server:
vllm serve --tensor-parallel-size=8 --async-scheduling --stream-interval 8 \
  --enable-chunked-prefill --speculative-config.method mtp \
  --speculative-config.num-speculative-tokens 1 --enable-prefix-caching \
  zai-org/GLM-4.7-FP8

# Run synthesis:
uv run --project prime-rl synthesize @ configs/synth.toml

# Manually upload outputs/ to HuggingFace as siro1/kernelbook-glm4_7-evals
```

## Environment Variables

Create a `.env` file with:
```bash
PRIME_API_KEY=your_prime_intellect_api_key
PRIME_TEAM_ID=your_team_id
HF_TOKEN=your_huggingface_token
DATASET_PATH=siro1/kernelbook-glm4_7-evals-filtered
FILTERED_DATASET_PATH=outputs/filtered_dataset.jsonl
UNIQUE_DATASET_PATH=outputs/filtered_dataset-filtered.jsonl
SYNTHETIC_DATASET_PATH=outputs/synthetic_prompts.jsonl
```

## Scripts

- **[filter_and_enrich_by_difficulty.py](scripts/filter_and_enrich_by_difficulty.py)** - Filter by reward and evaluate difficulty
- **[generate_prompts.py](scripts/generate_prompts.py)** - Generate synthetic task specifications
- **[filter_unique_best.py](scripts/filter_unique_best.py)** - Deduplicate by module name
- **[analyze_seq_length.py](scripts/analyze_seq_length.py)** - Analyze sequence lengths
- **[remove_reasoning.py](scripts/remove_reasoning.py)** - Remove reasoning from completions (keep answer only)
- **[upload_datasets.py](scripts/upload_datasets.py)** - Upload to HuggingFace with splits

See **[scripts/README.md](scripts/README.md)** for detailed documentation.

## Difficulty Ratings

- **low**: Trivial copy/identity kernel, simple elementwise operations
- **medium**: Basic reductions, standard matrix operations
- **high**: Fused operations, complex attention mechanisms, full model architectures

## Requirements

- Python 3.13+
- [uv](https://github.com/astral-sh/uv) package manager
- Prime Intellect API key (GPT-5.2 access)
- HuggingFace token
- [prime-rl](https://github.com/S1ro1/prime-rl) + vLLM (optional, for step 0 only)

## License

MIT
