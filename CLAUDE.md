# CLAUDE.md - Project Instructions

This project filters and evaluates the `siro1/kernelbook-glm4-evals` dataset for SFT training on Triton kernel code.

## Project Structure

```
dp-filtering/
├── scripts/           # Python scripts
│   ├── filter_dataset.py      # Main filtering pipeline
│   ├── analyze_correlation.py # Correlation analysis
│   └── keep_best.py           # Deduplication script
├── outputs/           # Generated datasets (gitignored)
│   ├── filtered_dataset.jsonl
│   └── filtered_dataset-filtered.jsonl
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
```python
from datasets import Dataset
from huggingface_hub import login
import json

login(token="your_token")
data = [json.loads(l) for l in open("outputs/filtered_dataset.jsonl")]
Dataset.from_list(data).push_to_hub("username/dataset-name")
```

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

- Filtered: https://huggingface.co/datasets/siro1/kernelbook-glm4-evals-filtered (7,181 samples)
- Unique: https://huggingface.co/datasets/siro1/kernelbook-glm4-evals-unique (2,967 samples)

## Notes

- All scripts should be run from the project root directory
- Large dataset files are gitignored - regenerate or download from HuggingFace
- The Prime Intellect API uses OpenAI-compatible endpoints at `https://api.pinference.ai/api/v1`
