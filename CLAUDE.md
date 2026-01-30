# CLAUDE.md - Project Instructions

This project processes kernel conversion datasets for SFT training on Triton kernel code.

## Project Structure

```
dp-data-pipeline/
├── scripts/                              # Python scripts
│   ├── config.py                            # Shared configuration (paths, HF upload)
│   ├── filter_and_enrich_by_difficulty.py   # Filter & evaluate difficulty
│   ├── generate_prompts.py                  # Synthetic prompt generation
│   ├── filter_unique_best.py                # Deduplication by module
│   ├── analyze_seq_length.py                # Sequence length analysis
│   └── remove_reasoning.py                  # Remove reasoning from completions
├── prime-rl/          # Submodule for synthesis pipeline
├── outputs/           # Generated datasets (gitignored)
│   └── {model}/       # Model-specific outputs (e.g., outputs/glm4_7/)
│       ├── filtered_dataset.jsonl
│       ├── unique_dataset.jsonl
│       └── synthetic_prompts.jsonl
├── .env               # API credentials (gitignored)
├── pyproject.toml     # Project dependencies
└── CLAUDE.md          # This file
```

## Quick Start

```bash
# Clone with submodules (for new clones)
git clone --recurse-submodules <repo-url>

# Or initialize submodules in existing clone
git submodule update --init --recursive

# Install dependencies
uv sync

# Set model name (optional, defaults to "glm4_7")
export DATA_MODEL=glm4_7

# Run complete pipeline (each script auto-uploads to HuggingFace)
uv run python scripts/filter_and_enrich_by_difficulty.py
uv run python scripts/generate_prompts.py
uv run python scripts/filter_unique_best.py
uv run python scripts/analyze_seq_length.py

# Optional: Remove reasoning from uploaded datasets
uv run python scripts/remove_reasoning.py filtered
uv run python scripts/remove_reasoning.py unique
```

## Environment Variables

Required in `.env`:
```
PRIME_API_KEY=your_prime_intellect_api_key
PRIME_TEAM_ID=your_team_id
HF_TOKEN=your_huggingface_token
```

Optional:
```
DATA_MODEL=glm4_7   # Model name for paths and repos (default: glm4_7)
TEST_MODE=true      # Run in test mode with limited samples
```

## Configuration

All scripts use `scripts/config.py` for shared configuration:

- **DATA_MODEL**: Environment variable that determines all paths and repo names
- **Local paths**: `outputs/{model}/filtered_dataset.jsonl`, etc.
- **HF repos**: `siro1/kernelbook-{model}-evals-filtered`, etc.

Example with different models:
```bash
# Process GLM-4.7 data (default)
DATA_MODEL=glm4_7 uv run python scripts/filter_and_enrich_by_difficulty.py

# Process Qwen data
DATA_MODEL=qwen3_30b uv run python scripts/filter_and_enrich_by_difficulty.py
```

## Workflow

### 0. Generate Base Dataset (Optional)
Generates the initial `siro1/kernelbook-{model}-evals` dataset using prime-rl synthesis pipeline.

**Prerequisites:**
- Initialize the prime-rl submodule (see Quick Start)
- Start vLLM inference server:
```bash
vllm serve --tensor-parallel-size=8 --async-scheduling --stream-interval 8 \
  --enable-chunked-prefill --speculative-config.method mtp \
  --speculative-config.num-speculative-tokens 1 --enable-prefix-caching \
  zai-org/GLM-4.7-FP8
```

**Run synthesis:**
```bash
uv run --project prime-rl synthesize @ configs/synth.toml
```

**Output:** Evaluation in `outputs/` directory (manually upload to HuggingFace as `siro1/kernelbook-{model}-evals`)

### 1. Filter and Enrich by Difficulty
Evaluates samples from `siro1/kernelbook-{model}-evals` with reward > 0.01 using GPT-5.2.
Adds difficulty rating (low/medium/high) to each sample.

```bash
# Full run with auto-upload
uv run python scripts/filter_and_enrich_by_difficulty.py

# Skip upload (local only)
uv run python scripts/filter_and_enrich_by_difficulty.py --no-upload

# Test mode (5 samples)
TEST_MODE=true uv run python scripts/filter_and_enrich_by_difficulty.py
```

Output: `outputs/{model}/filtered_dataset.jsonl` with all original columns + `difficulty`, `evaluation_raw`
Uploads to: `siro1/kernelbook-{model}-evals-filtered`

### 2. Generate Synthetic Prompts
Generates task specifications from PyTorch modules using kimi-k2-0905. Creates (prompt, module_name, python_code) tuples for SFT training.

```bash
# Full run with auto-upload
uv run python scripts/generate_prompts.py

# Skip upload
uv run python scripts/generate_prompts.py --no-upload

# Test mode (10 samples)
TEST_MODE=true uv run python scripts/generate_prompts.py
```

Output: `outputs/{model}/synthetic_prompts.jsonl`
Uploads to: `siro1/kernelbook-{model}-synthetic-tasks`

### 3. Deduplicate by Module
Keeps only the best sample (highest reward) per module name.

```bash
# Full run with auto-upload
uv run python scripts/filter_unique_best.py

# Skip upload
uv run python scripts/filter_unique_best.py --no-upload
```

Output: `outputs/{model}/unique_dataset.jsonl`
Uploads to: `siro1/kernelbook-{model}-evals-unique`

### 4. Analyze Sequence Lengths
Analyzes token sequence lengths to determine optimal `seq_len` for SFT training.

```bash
# Analyze filtered dataset (default)
uv run python scripts/analyze_seq_length.py

# Analyze unique dataset
uv run python scripts/analyze_seq_length.py --unique

# Analyze source evals dataset
uv run python scripts/analyze_seq_length.py --source
```

Provides statistics and recommendations for training configuration.

### 5. Remove Reasoning (Optional)
Removes reasoning from completions, keeping only the answer field.

```bash
# Process any dataset (source/filtered/unique)
uv run python scripts/remove_reasoning.py source
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

All repos follow the pattern `siro1/kernelbook-{model}-...`:

- **Source:** https://huggingface.co/datasets/GPUMODE/KernelBook (18,162 samples) - Original PyTorch modules
- **Base:** `siro1/kernelbook-{model}-evals` - Generated using prime-rl synthesis
- **Filtered:** `siro1/kernelbook-{model}-evals-filtered` (90/10 split) - Reward > 0.01 with difficulty ratings
- **Unique:** `siro1/kernelbook-{model}-evals-unique` (90/10 split) - Deduplicated by module
- **Synthetic:** `siro1/kernelbook-{model}-synthetic-tasks` - Task specifications
- **Filtered (No Reasoning):** `siro1/kernelbook-{model}-evals-filtered-no-reasoning` (90/10 split)
- **Unique (No Reasoning):** `siro1/kernelbook-{model}-evals-unique-no-reasoning` (90/10 split)

## Submodule Management

The `prime-rl` directory is a git submodule. Common operations:

```bash
# Update submodule to latest commit from remote
git submodule update --remote prime-rl

# After pulling changes that updated the submodule reference
git submodule update --init --recursive

# Check submodule status
git submodule status
```

## Notes

- All scripts should be run from the project root directory
- Large dataset files are gitignored - regenerate or download from HuggingFace
- The Prime Intellect API uses OpenAI-compatible endpoints at `https://api.pinference.ai/api/v1`
- Base dataset generation (step 0) requires vLLM
- Step 0 is optional - you can start from existing datasets on HuggingFace
- All processing scripts auto-upload to HuggingFace by default (use `--no-upload` to skip)

## Code Rules
- Do not write code that can silently produce wrong results, instead code should explicitly fail
- Never do hacks, that can silently cause wrong results, i.e. truncating code inside of prompts, etc.
- Write code that is easy to understand, if section of code is not straightforward, add comments
