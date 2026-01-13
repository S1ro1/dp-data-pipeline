# Plots

Visualization outputs from correlation analysis.

## Files

### `correlation_analysis.png`
6-panel analysis of model evaluation vs benchmark reward:
1. **Difficulty vs Reward** scatter with regression line
2. **Style vs Reward** scatter with regression line
3. **Difficulty vs Style** scatter with regression line
4. **Reward Distribution by Difficulty** box plot
5. **Reward Distribution by Style** box plot
6. **Mean Reward Heatmap** by difficulty and style

### `joint_distribution.png`
Scatter plot showing joint distribution of difficulty, style, and reward (color-coded).

## Key Findings

### Correlations with Benchmark Reward

| Metric | Pearson r | Spearman ρ | Interpretation |
|--------|-----------|------------|----------------|
| Difficulty | +0.066 | +0.114 | Negligible/weak positive |
| Style | -0.034 | -0.067 | Negligible negative |

### Cross-correlation
- **Difficulty vs Style**: r = -0.49 (moderate negative)
  - Harder tasks tend to have lower style scores
  - May indicate evaluation bias or genuine complexity trade-off

### Conclusions
1. Benchmark reward is largely independent of subjective difficulty/style ratings
2. The model rates harder kernel conversions as having lower code quality
3. Style evaluation needs further calibration for better distribution

## Reproduction

```bash
uv run python scripts/analyze_correlation.py
```

Requires `outputs/filtered_dataset.jsonl` to exist.
