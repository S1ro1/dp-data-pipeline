# Scripts Documentation

Detailed documentation for the data processing pipeline scripts.

## Configuration

All scripts use `config.py` for shared configuration. The key environment variable is:

- **`DATA_MODEL`**: Determines all paths and HuggingFace repo names (default: `glm4_7`)

### Path Patterns

| Type | Pattern |
|------|---------|
| Local files | `outputs/{model}/filtered_dataset.jsonl` |
| HF repos | `siro1/kernelbook-{model}-evals-filtered` |

### Example

```bash
# Process GLM-4.7 data (default)
DATA_MODEL=glm4_7 uv run python scripts/filter_and_enrich_by_difficulty.py

# Process Qwen data
DATA_MODEL=qwen3_30b uv run python scripts/filter_and_enrich_by_difficulty.py
```

## Pipeline Overview

0. **Generate Base Dataset** (Optional) - Create evaluation dataset using vf-eval
0.5. **Convert to Evals Format** - Convert vf-eval output to base evals format → auto-uploads
1. **Filter & Enrich** - Evaluate samples with difficulty ratings → auto-uploads
2. **Generate Prompts** - Create synthetic task specifications → auto-uploads
3. **Deduplicate** - Keep best samples per module → auto-uploads
4. **Analyze Lengths** - Determine optimal sequence lengths
5. **Remove Reasoning** (Optional) - Strip reasoning from uploaded datasets

---

## Step 0: Generate Base Dataset (Optional)

**Prerequisites:**
- Start vLLM inference server with reasoning model

**Commands:**
```bash
# Terminal 1: Start vLLM server
vllm serve moonshotai/Kimi-K2-Thinking --tensor-parallel-size 8 \
  --decode-context-parallel-size 8 --enable-auto-tool-choice \
  --tool-call-parser kimi_k2 --reasoning-parser kimi_k2 --trust-remote-code \
  --async-scheduling --stream-interval 8 --enable-chunked-prefill \
  --no-enable-prefix-caching

# Terminal 2: Run synthesis
vf-eval siro/kernelbook-env --model moonshotai/Kimi-K2-Thinking \
  --api-base-url http://localhost:8000/v1 --num-examples -1 \
  --rollouts-per-example=1 --max-concurrent 512 -s -C trajectory -R --tui
```

**Output:** Evaluation results in `outputs/` directory as `kernelbook-env.jsonl`

**Note:** This step is optional - you can start from existing datasets on HuggingFace.

---

## Step 0.5: `convert_generation_to_evals.py`

Converts vf-eval output (with trajectory data) to the base evals format for downstream processing.

### What it does
- Reads vf-eval output JSONL file (contains trajectory with reasoning)
- Extracts `reasoning_content` from `trajectory[-1].response.choices[0].message.reasoning_content`
- Adds reasoning to completion: `[{"role": "assistant", "content": "...", "reasoning": "..."}]`
- Removes trajectory field (large and not needed downstream)
- Saves to `outputs/{model}/evals_dataset.jsonl`
- Auto-uploads to `siro1/kernelbook-{model}-evals`

### Usage
```bash
# Convert and upload to HuggingFace (default)
DATA_MODEL=kimi_k2_thinking uv run python scripts/convert_generation_to_evals.py outputs/kernelbook-env.jsonl

# Convert locally only (skip upload)
uv run python scripts/convert_generation_to_evals.py outputs/kernelbook-env.jsonl --no-upload

# Custom output path
uv run python scripts/convert_generation_to_evals.py input.jsonl -o custom_output.jsonl
```

### Input Format (vf-eval output)
```json
{
  "example_id": 0,
  "prompt": [{"role": "user", "content": "..."}],
  "completion": [{"role": "assistant", "content": "..."}],
  "answer": "extracted code",
  "task": "default",
  "info": {"module_name": "...", "python_code": "...", "triton_code": "..."},
  "reward": 0.5,
  "generation_ms": 1000.0,
  "scoring_ms": 500.0,
  "total_ms": 1500.0,
  "speedup_reward": 0.5,
  "num_turns": 1.0,
  "trajectory": [{
    "response": {
      "choices": [{"message": {"reasoning_content": "...thinking..."}}]
    }
  }]
}
```

### Output Format (base evals)
```json
{
  "example_id": 0,
  "prompt": [{"role": "user", "content": "..."}],
  "completion": [{"role": "assistant", "content": "...", "reasoning": "...thinking..."}],
  "task": "default",
  "reward": 0.5,
  "generation_ms": 1000.0,
  "scoring_ms": 500.0,
  "total_ms": 1500.0,
  "info": {"module_name": "...", "python_code": "...", "triton_code": "..."},
  "answer": "extracted code",
  "speedup_reward": 0.5,
  "num_turns": 1.0,
  "oai_tools": null
}
```

---

## Step 1: `filter_and_enrich_by_difficulty.py`

Filters high-quality samples and enriches them with difficulty ratings using GPT-5.2.

### What it does
- Loads `siro1/kernelbook-{model}-evals` from HuggingFace
- Filters samples with `reward > 0.85`
- Evaluates each sample for difficulty: **low**, **medium**, or **high**
- Preserves all original columns + adds `difficulty`, `evaluation_raw`
- Saves to `outputs/{model}/filtered_dataset.jsonl`
- Auto-uploads to `siro1/kernelbook-{model}-evals-filtered`

### Usage
```bash
# Full run with auto-upload
uv run python scripts/filter_and_enrich_by_difficulty.py

# Skip upload (local only)
uv run python scripts/filter_and_enrich_by_difficulty.py --no-upload

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

Generates synthetic task specifications from PyTorch modules using kimi-k2-0905.

### What it does
- Loads `GPUMODE/KernelBook` dataset (18,162 samples)
- Generates detailed task specifications describing what to implement
- Produces `(prompt, module_name, python_code, triton_code, uuid)` tuples
- Does NOT include Triton implementation details or code snippets
- Saves to `outputs/{model}/synthetic_prompts.jsonl`
- Auto-uploads to `siro1/kernelbook-{model}-synthetic-tasks`

### Usage
```bash
# Full run with auto-upload
uv run python scripts/generate_prompts.py

# Skip upload
uv run python scripts/generate_prompts.py --no-upload

# Test mode (10 samples)
TEST_MODE=true uv run python scripts/generate_prompts.py
```

### Output Format
```json
{
  "prompt": "Detailed task specification...",
  "module_name": "LayerNorm",
  "python_code": "class LayerNorm(nn.Module):...",
  "triton_code": "...",
  "uuid": "..."
}
```

### Configuration
- `BATCH_SIZE = 1024` - concurrent API requests
- `MODEL = "moonshotai/kimi-k2-0905"` - generation model
- `MAX_RETRIES = 3` - retry attempts per sample

---

## Step 3: `filter_unique_best.py`

Deduplicates the filtered dataset by keeping only the best sample per module name.

### What it does
- Reads `outputs/{model}/filtered_dataset.jsonl`
- Groups samples by `module_name`
- Keeps the sample with highest `reward` per module
- Saves to `outputs/{model}/unique_dataset.jsonl`
- Auto-uploads to `siro1/kernelbook-{model}-evals-unique`

### Usage
```bash
# Full run with auto-upload
uv run python scripts/filter_unique_best.py

# Skip upload
uv run python scripts/filter_unique_best.py --no-upload
```

---

## Step 4: `analyze_seq_length.py`

Analyzes sequence lengths in the dataset to determine optimal `seq_len` for SFT training.

### What it does
- Loads dataset from HuggingFace (filtered by default)
- Tokenizes all samples using Qwen tokenizer with chat template
- Computes sequence length statistics and percentiles
- Recommends optimal `seq_len` for 99%, 99.5%, and 100% coverage

### Usage
```bash
# Analyze filtered dataset (default)
uv run python scripts/analyze_seq_length.py

# Analyze unique dataset
uv run python scripts/analyze_seq_length.py --unique

# Analyze source evals dataset
uv run python scripts/analyze_seq_length.py --source
```

### Configuration
- `MODEL_NAME = "Qwen/Qwen3-30B-A3B-Thinking-2507"` - tokenizer model

### Output
- Sequence length distribution by bucket
- Percentile statistics (50%, 75%, 90%, 95%, 99%, 99.5%, 99.9%, 100%)
- Recommended `seq_len` values for different coverage levels
- Count of samples exceeding common limits

---

## Step 5: `remove_reasoning.py` (Optional)

Removes reasoning from completion content, keeping only the answer field.

### What it does
- Loads a dataset from HuggingFace (source, filtered, or unique)
- Parses `<answer>...</answer>` from `completion[0]["content"]`
- Removes all reasoning, keeping only the answer wrapped in tags
- Preserves dataset splits (train/validation) if they exist
- Uploads result with `-no-reasoning` suffix

### Usage
```bash
# Process source dataset
uv run python scripts/remove_reasoning.py source

# Process filtered dataset
uv run python scripts/remove_reasoning.py filtered

# Process unique dataset
uv run python scripts/remove_reasoning.py unique
```

### Output Datasets
- `siro1/kernelbook-{model}-evals-no-reasoning`
- `siro1/kernelbook-{model}-evals-filtered-no-reasoning`
- `siro1/kernelbook-{model}-evals-unique-no-reasoning`

### Use Case
Use these datasets when you want to train models without chain-of-thought reasoning, providing only the final answer in the completion.

---

## Complete Pipeline

Run the full pipeline from scratch:

```bash
# 1. Install dependencies
uv sync

# 2. Configure .env
cat > .env << EOF
PRIME_API_KEY=your_prime_intellect_api_key
PRIME_TEAM_ID=your_team_id
HF_TOKEN=your_huggingface_token
EOF

# 3. Set model name
export DATA_MODEL=kimi_k2_thinking

# 4. (Optional) Generate base dataset with vf-eval
# See Step 0 for vLLM and vf-eval commands

# 5. (Optional) Convert vf-eval output to evals format
uv run python scripts/convert_generation_to_evals.py outputs/kernelbook-env.jsonl

# 6. Run pipeline (each script auto-uploads)
uv run python scripts/filter_and_enrich_by_difficulty.py
uv run python scripts/generate_prompts.py
uv run python scripts/filter_unique_best.py
uv run python scripts/analyze_seq_length.py

# 7. (Optional) Remove reasoning from uploaded datasets
uv run python scripts/remove_reasoning.py filtered
uv run python scripts/remove_reasoning.py unique
```

## Test Mode

Scripts support `TEST_MODE` environment variable to run on a small subset:

```bash
TEST_MODE=true uv run python scripts/filter_and_enrich_by_difficulty.py
TEST_MODE=true uv run python scripts/generate_prompts.py
```

## Skipping Upload

All scripts that produce output support `--no-upload` to skip HuggingFace upload:

```bash
uv run python scripts/filter_and_enrich_by_difficulty.py --no-upload
uv run python scripts/generate_prompts.py --no-upload
uv run python scripts/filter_unique_best.py --no-upload
```
