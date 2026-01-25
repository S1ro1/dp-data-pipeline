# Scripts Documentation

Detailed documentation for the data processing pipeline scripts.

## Pipeline Overview

0. **Generate Base Dataset** (Optional) - Create evaluation dataset using prime-rl
1. **Filter & Enrich** - Evaluate samples with difficulty ratings
2. **Generate Prompts** - Create synthetic task specifications
3. **Deduplicate** - Keep best samples per module
4. **Analyze Lengths** - Determine optimal sequence lengths
5. **Upload** - Push datasets to HuggingFace

---

## Step 0: Generate Base Dataset (Optional)

**Prerequisites:**
- Install [prime-rl](https://github.com/S1ro1/prime-rl)
- Start vLLM inference server

**Commands:**
```bash
# Terminal 1: Start vLLM server
vllm serve --tensor-parallel-size=8 --async-scheduling --stream-interval 8 \
  --enable-chunked-prefill --speculative-config.method mtp \
  --speculative-config.num-speculative-tokens 1 --enable-prefix-caching \
  zai-org/GLM-4.7-FP8

# Terminal 2: Run synthesis
uv run synthesize @ configs/synth.toml
```

**Output:** Evaluation results in `outputs/` directory (manually upload to HuggingFace as `siro1/kernelbook-glm4_7-evals`)

**Note:** This step is optional - you can start from the existing dataset on HuggingFace.

---

## Step 1: `filter_and_enrich_by_difficulty.py`

Filters high-quality samples and enriches them with difficulty ratings using GPT-5.2.

### What it does
- Loads `siro1/kernelbook-glm4_7-evals` from HuggingFace
- Filters samples with `reward > 0.85` (~7,181 out of 18,162)
- Evaluates each sample for difficulty: **low**, **medium**, or **high**
- Preserves all original columns + adds `difficulty`, `evaluation_raw`
- Outputs to `outputs/filtered_dataset.jsonl`

### Usage
```bash
# Full run (~7,181 samples)
uv run python scripts/filter_and_enrich_by_difficulty.py

# Test mode (5 samples)
TEST_MODE=true uv run python scripts/filter_and_enrich_by_difficulty.py
```

### Configuration
- `BATCH_SIZE = 256` - concurrent API requests
- `REWARD_THRESHOLD = 0.85` - minimum reward to include
- `MODEL = "openai/gpt-5.2"` - evaluation model via Prime Intellect API

### Difficulty Criteria
- **low**: Trivial copy/identity kernel, elementwise operations
- **medium**: Basic reductions, matrix operations
- **high**: Fused operations, full model architectures

---

## Step 2: `generate_prompts.py`

Generates synthetic task specifications from PyTorch modules using GPT-5.2.

### What it does
- Loads `GPUMODE/KernelBook` dataset (18,162 samples)
- Generates detailed task specifications describing what to implement
- Produces `(prompt, module_name, python_code)` tuples
- Does NOT include Triton implementation details or code snippets
- Outputs to `outputs/synthetic_prompts.jsonl`

### Usage
```bash
# Full run (18,162 samples)
uv run python scripts/generate_prompts.py

# Test mode (5 samples)
TEST_MODE=true uv run python scripts/generate_prompts.py
```

### Output Format
```json
{
  "prompt": "Detailed task specification...",
  "module_name": "LayerNorm",
  "python_code": "class LayerNorm(nn.Module):..."
}
```

### Configuration
- `BATCH_SIZE = 256` - concurrent API requests
- `MODEL = "openai/gpt-5.2"` - generation model
- `MAX_RETRIES = 3` - retry attempts per sample

---

## Step 3: `filter_unique_best.py`

Deduplicates the filtered dataset by keeping only the best sample per module name.

### What it does
- Reads `outputs/filtered_dataset.jsonl`
- Groups samples by `module_name`
- Keeps the sample with highest `reward` per module
- Outputs to `outputs/filtered_dataset-filtered.jsonl`

### Usage
```bash
uv run python scripts/filter_unique_best.py
```

### Statistics
- Input: ~7,181 samples
- Output: ~2,967 unique modules
- Reduction: ~58.7%

---

## Step 4: `analyze_seq_length.py`

Analyzes sequence lengths in the dataset to determine optimal `seq_len` for SFT training.

### What it does
- Loads dataset from `DATASET_PATH` environment variable
- Tokenizes all samples using Qwen tokenizer with chat template
- Computes sequence length statistics and percentiles
- Recommends optimal `seq_len` for 99%, 99.5%, and 100% coverage
- Helps prevent truncation during training

### Usage
```bash
uv run python scripts/analyze_seq_length.py
```

### Configuration
- `MODEL_NAME = "Qwen/Qwen3-30B-A3B-Thinking-2507"` - tokenizer model
- `DATASET_PATH` - environment variable (e.g., `siro1/kernelbook-glm4_7-evals-filtered`)

### Output
- Sequence length distribution by bucket
- Percentile statistics (50%, 75%, 90%, 95%, 99%, 99.5%, 99.9%, 100%)
- Recommended `seq_len` values for different coverage levels
- Count of samples exceeding common limits

---

## Step 5: `upload_datasets.py`

Uploads processed datasets to HuggingFace with train/validation splits.

### What it does
- Creates 90/10 train/validation splits (filtered and unique datasets)
- Uploads to HuggingFace Hub with proper metadata
- Supports selective upload of individual datasets

### Usage
```bash
# Upload all datasets
uv run python scripts/upload_datasets.py all

# Upload specific datasets
uv run python scripts/upload_datasets.py filtered
uv run python scripts/upload_datasets.py unique
uv run python scripts/upload_datasets.py synthetic
```

### Environment Variables Required
- `HF_TOKEN` - HuggingFace authentication token
- `FILTERED_DATASET_PATH` - path to filtered dataset JSONL
- `UNIQUE_DATASET_PATH` - path to unique dataset JSONL
- `SYNTHETIC_DATASET_PATH` - path to synthetic prompts JSONL

### Uploaded Datasets
- `siro1/kernelbook-glm4_7-evals-filtered` - with 90/10 train/val split
- `siro1/kernelbook-glm4_7-evals-unique` - with 90/10 train/val split
- `siro1/kernelbook-synthetic-tasks` - no split (single train)

---

## Complete Pipeline

Run the full pipeline from scratch:

```bash
# 0. (Optional) Generate base dataset
# Requires prime-rl + vLLM - see Step 0 above

# 1. Install dependencies
uv sync

# 2. Configure .env
cat > .env << EOF
PRIME_API_KEY=your_prime_intellect_api_key
PRIME_TEAM_ID=your_team_id
HF_TOKEN=your_huggingface_token
DATASET_PATH=siro1/kernelbook-glm4_7-evals-filtered
FILTERED_DATASET_PATH=outputs/filtered_dataset.jsonl
UNIQUE_DATASET_PATH=outputs/filtered_dataset-filtered.jsonl
SYNTHETIC_DATASET_PATH=outputs/synthetic_prompts.jsonl
EOF

# 3. Run pipeline
uv run python scripts/filter_and_enrich_by_difficulty.py
uv run python scripts/generate_prompts.py
uv run python scripts/filter_unique_best.py
uv run python scripts/analyze_seq_length.py
uv run python scripts/upload_datasets.py all
```

## Test Mode

All batch processing scripts support `TEST_MODE` environment variable to run on a small subset (5 samples):

```bash
TEST_MODE=true uv run python scripts/filter_and_enrich_by_difficulty.py
TEST_MODE=true uv run python scripts/generate_prompts.py
```
