# CLAUDE.md - Project Instructions

This project filters and evaluates the `siro1/kernelbook-glm4-evals` dataset for SFT training on Triton kernel code.

## Project Structure

```
dp-filtering/
├── scripts/           # Python scripts
│   ├── filter_dataset.py      # Main filtering pipeline
│   ├── generate_prompts.py    # Synthetic prompt generation
│   ├── analyze_correlation.py # Correlation analysis
│   ├── keep_best.py           # Deduplication script
│   └── upload_datasets.py     # Upload to HuggingFace with splits
├── outputs/           # Generated datasets (gitignored)
│   ├── filtered_dataset.jsonl
│   ├── filtered_dataset-filtered.jsonl
│   └── synthetic_prompts.jsonl
├── plots/             # Visualization outputs
│   ├── correlation_analysis.png
│   └── joint_distribution.png
├── .env               # API credentials (gitignored)
├── pyproject.toml     # Project dependencies
└── CLAUDE.md          # This file
```

## Quick Start

```bash
# Install dependencies
uv sync

# Run filtering (requires .env with API keys)
uv run python scripts/filter_dataset.py

# Run analysis
uv run python scripts/analyze_correlation.py
```

## Environment Variables

Required in `.env`:
```
PRIME_API_KEY=your_prime_intellect_api_key
PRIME_TEAM_ID=your_team_id
HF_TOKEN=your_huggingface_token
```

## Workflow

### 1. Filter Dataset
Evaluates samples from `siro1/kernelbook-glm4-evals` with reward > 0.85 using GPT-5.2.

```bash
# Full run (7,181 samples, ~3-5 min)
uv run python scripts/filter_dataset.py

# Test mode (5 samples)
TEST_MODE=true uv run python scripts/filter_dataset.py
```

### 2. Analyze Correlations
Generates statistics and plots for difficulty/style vs reward correlations.

```bash
uv run python scripts/analyze_correlation.py
```

### 3. Upload to HuggingFace
Uploads both datasets with 90/10 train/validation splits.

```bash
uv run python scripts/upload_datasets.py
```

### 4. Generate Synthetic Prompts
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

## Evaluation Criteria

### Difficulty (0-10)
- 0: Trivial copy/identity kernel
- 1-2: Simple elementwise (relu, add)
- 3-4: Basic reductions
- 5-6: Moderate (softmax, layernorm)
- 7-8: Complex (attention, conv2d)
- 9-10: Advanced (FlashAttention, ResNet)

### Style (0-10)
- 0-1: Broken/unreadable
- 2-3: Messy but works
- 4-5: Acceptable/functional
- 5-6: Good structure
- 7-8: Very good (efficient patterns)
- 9-10: Excellent (optimal, educational)

## Key Findings

- **Difficulty-Reward**: Weak positive correlation (r ≈ +0.07)
- **Style-Reward**: Negligible correlation (r ≈ -0.03)
- **Difficulty-Style**: Moderate negative (r ≈ -0.49)
- Benchmark reward is largely independent of subjective evaluations

## HuggingFace Datasets

- Original: https://huggingface.co/datasets/siro1/kernelbook-glm4-evals (18,162 samples: 16,345 train / 1,817 val)
- Filtered: https://huggingface.co/datasets/siro1/kernelbook-glm4-evals-filtered (7,181 samples: 6,462 train / 719 val)
- Unique: https://huggingface.co/datasets/siro1/kernelbook-glm4-evals-unique (2,967 samples: 2,670 train / 297 val)

## Notes

- All scripts should be run from the project root directory
- Large dataset files are gitignored - regenerate or download from HuggingFace
- The Prime Intellect API uses OpenAI-compatible endpoints at `https://api.pinference.ai/api/v1`

## Code Rules
- Do not write code that can silently produce wrong results, instead code should explicitly fail
- Never do hacks, that can silently cause wrong results, i.e. truncating code inside of prompts, etc.
- Write code that is easy to understand, if section of code is not straightforward, add comments
