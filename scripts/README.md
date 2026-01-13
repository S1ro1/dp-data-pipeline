# Scripts

Python scripts for dataset filtering and analysis.

## Files

### `filter_dataset.py`
Batched filtering script that evaluates kernel conversion samples using GPT-5.2 via Prime Intellect API.

**What it does:**
- Loads `siro1/kernelbook-glm4-evals` dataset from HuggingFace
- Filters samples with `reward > 0.85` (7,181 out of 18,162)
- Evaluates each sample for difficulty (0-10) and style quality (0-10)
- Preserves all original columns + adds `difficulty`, `style`, `evaluation_raw`
- Outputs to `outputs/filtered_dataset.jsonl`

**Usage:**
```bash
# Full run
uv run python scripts/filter_dataset.py

# Test mode (5 samples)
TEST_MODE=true uv run python scripts/filter_dataset.py
```

**Configuration (in script):**
- `BATCH_SIZE = 256` - concurrent API requests
- `REWARD_THRESHOLD = 0.85` - minimum reward to include
- `MODEL = "openai/gpt-5.2"` - evaluation model

### `analyze_correlation.py`
Analyzes correlation between model evaluation (difficulty/style) and benchmark reward.

**What it does:**
- Computes Pearson and Spearman correlations
- Generates distribution statistics
- Creates visualization plots saved to `plots/`

**Usage:**
```bash
uv run python scripts/analyze_correlation.py
```

**Output:**
- `plots/correlation_analysis.png` - 6-panel correlation analysis
- `plots/joint_distribution.png` - joint distribution scatter plot

### `keep_best.py`
Deduplicates the filtered dataset by module name, keeping unique samples.

**Usage:**
```bash
uv run python scripts/keep_best.py
```

## Reproduction

1. Set up environment:
```bash
uv sync
```

2. Add credentials to `.env`:
```
PRIME_API_KEY=your_key
PRIME_TEAM_ID=your_team_id
HF_TOKEN=your_hf_token
```

3. Run filtering:
```bash
uv run python scripts/filter_dataset.py
```

4. Run analysis:
```bash
uv run python scripts/analyze_correlation.py
```
