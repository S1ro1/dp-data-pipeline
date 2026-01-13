#!/usr/bin/env python3
"""
Analyze correlation between model evaluation (difficulty/style) and benchmark reward.
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy import stats

# Load data
data = []
with open("outputs/filtered_dataset.jsonl", "r") as f:
    for line in f:
        data.append(json.loads(line))

print(f"Loaded {len(data)} samples")

# Extract relevant fields
rewards = []
difficulties = []
styles = []

for sample in data:
    reward = sample.get("reward")
    difficulty = sample.get("difficulty")
    style = sample.get("style")

    if reward is not None and difficulty is not None and style is not None:
        rewards.append(reward)
        difficulties.append(difficulty)
        styles.append(style)

rewards = np.array(rewards)
difficulties = np.array(difficulties)
styles = np.array(styles)

print(f"Samples with all fields: {len(rewards)}")

# Basic statistics
print("\n" + "=" * 60)
print("BASIC STATISTICS")
print("=" * 60)
print(f"\nReward:")
print(f"  Mean: {rewards.mean():.4f}, Std: {rewards.std():.4f}")
print(f"  Min: {rewards.min():.4f}, Max: {rewards.max():.4f}")

print(f"\nDifficulty:")
print(f"  Mean: {difficulties.mean():.2f}, Std: {difficulties.std():.2f}")
print(f"  Min: {difficulties.min()}, Max: {difficulties.max()}")

print(f"\nStyle:")
print(f"  Mean: {styles.mean():.2f}, Std: {styles.std():.2f}")
print(f"  Min: {styles.min()}, Max: {styles.max()}")

# Correlation analysis
print("\n" + "=" * 60)
print("CORRELATION ANALYSIS")
print("=" * 60)

# Pearson correlation
r_diff, p_diff = stats.pearsonr(difficulties, rewards)
r_style, p_style = stats.pearsonr(styles, rewards)
r_diff_style, p_diff_style = stats.pearsonr(difficulties, styles)

print(f"\nPearson Correlations:")
print(f"  Difficulty vs Reward:  r = {r_diff:+.4f}, p = {p_diff:.2e}")
print(f"  Style vs Reward:       r = {r_style:+.4f}, p = {p_style:.2e}")
print(f"  Difficulty vs Style:   r = {r_diff_style:+.4f}, p = {p_diff_style:.2e}")

# Spearman correlation (rank-based, better for non-linear relationships)
rho_diff, p_diff_s = stats.spearmanr(difficulties, rewards)
rho_style, p_style_s = stats.spearmanr(styles, rewards)
rho_diff_style, p_diff_style_s = stats.spearmanr(difficulties, styles)

print(f"\nSpearman Correlations:")
print(f"  Difficulty vs Reward:  rho = {rho_diff:+.4f}, p = {p_diff_s:.2e}")
print(f"  Style vs Reward:       rho = {rho_style:+.4f}, p = {p_style_s:.2e}")
print(f"  Difficulty vs Style:   rho = {rho_diff_style:+.4f}, p = {p_diff_style_s:.2e}")

# Reward by difficulty level
print("\n" + "=" * 60)
print("REWARD BY DIFFICULTY LEVEL")
print("=" * 60)
for d in range(11):
    mask = difficulties == d
    if mask.sum() > 0:
        r_mean = rewards[mask].mean()
        r_std = rewards[mask].std()
        count = mask.sum()
        bar = "█" * int(r_mean * 20)
        print(f"  Difficulty {d:2d}: n={count:4d}, reward={r_mean:.4f} ± {r_std:.4f} {bar}")

# Reward by style level
print("\n" + "=" * 60)
print("REWARD BY STYLE LEVEL")
print("=" * 60)
for s in range(11):
    mask = styles == s
    if mask.sum() > 0:
        r_mean = rewards[mask].mean()
        r_std = rewards[mask].std()
        count = mask.sum()
        bar = "█" * int(r_mean * 20)
        print(f"  Style {s:2d}: n={count:4d}, reward={r_mean:.4f} ± {r_std:.4f} {bar}")

# Create visualizations
print("\n" + "=" * 60)
print("GENERATING VISUALIZATIONS")
print("=" * 60)

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle("Correlation Analysis: Model Evaluation vs Benchmark Reward", fontsize=14)

# 1. Difficulty vs Reward scatter
ax = axes[0, 0]
ax.scatter(difficulties, rewards, alpha=0.3, s=10)
z = np.polyfit(difficulties, rewards, 1)
p = np.poly1d(z)
x_line = np.linspace(difficulties.min(), difficulties.max(), 100)
ax.plot(x_line, p(x_line), "r-", linewidth=2, label=f"r={r_diff:.3f}")
ax.set_xlabel("Difficulty")
ax.set_ylabel("Reward")
ax.set_title("Difficulty vs Reward")
ax.legend()

# 2. Style vs Reward scatter
ax = axes[0, 1]
ax.scatter(styles, rewards, alpha=0.3, s=10)
z = np.polyfit(styles, rewards, 1)
p = np.poly1d(z)
x_line = np.linspace(styles.min(), styles.max(), 100)
ax.plot(x_line, p(x_line), "r-", linewidth=2, label=f"r={r_style:.3f}")
ax.set_xlabel("Style")
ax.set_ylabel("Reward")
ax.set_title("Style vs Reward")
ax.legend()

# 3. Difficulty vs Style scatter
ax = axes[0, 2]
ax.scatter(difficulties, styles, alpha=0.3, s=10)
z = np.polyfit(difficulties, styles, 1)
p = np.poly1d(z)
x_line = np.linspace(difficulties.min(), difficulties.max(), 100)
ax.plot(x_line, p(x_line), "r-", linewidth=2, label=f"r={r_diff_style:.3f}")
ax.set_xlabel("Difficulty")
ax.set_ylabel("Style")
ax.set_title("Difficulty vs Style")
ax.legend()

# 4. Box plot: Reward by Difficulty
ax = axes[1, 0]
difficulty_groups = [rewards[difficulties == d] for d in range(11) if (difficulties == d).sum() > 0]
difficulty_labels = [str(d) for d in range(11) if (difficulties == d).sum() > 0]
bp = ax.boxplot(difficulty_groups, labels=difficulty_labels, patch_artist=True)
for patch in bp['boxes']:
    patch.set_facecolor('lightblue')
ax.set_xlabel("Difficulty")
ax.set_ylabel("Reward")
ax.set_title("Reward Distribution by Difficulty")

# 5. Box plot: Reward by Style
ax = axes[1, 1]
style_groups = [rewards[styles == s] for s in range(11) if (styles == s).sum() > 0]
style_labels = [str(s) for s in range(11) if (styles == s).sum() > 0]
bp = ax.boxplot(style_groups, labels=style_labels, patch_artist=True)
for patch in bp['boxes']:
    patch.set_facecolor('lightgreen')
ax.set_xlabel("Style")
ax.set_ylabel("Reward")
ax.set_title("Reward Distribution by Style")

# 6. Heatmap: Mean reward by difficulty and style
ax = axes[1, 2]
heatmap_data = np.zeros((11, 11))
heatmap_counts = np.zeros((11, 11))
for d, s, r in zip(difficulties, styles, rewards):
    heatmap_data[d, s] += r
    heatmap_counts[d, s] += 1

# Avoid division by zero
heatmap_counts[heatmap_counts == 0] = np.nan
heatmap_mean = heatmap_data / heatmap_counts

# Only show rows/cols with data
valid_rows = ~np.all(np.isnan(heatmap_mean), axis=1)
valid_cols = ~np.all(np.isnan(heatmap_mean), axis=0)
heatmap_trimmed = heatmap_mean[valid_rows][:, valid_cols]
row_labels = [str(i) for i in range(11) if valid_rows[i]]
col_labels = [str(i) for i in range(11) if valid_cols[i]]

sns.heatmap(heatmap_trimmed, ax=ax, cmap="YlOrRd", annot=True, fmt=".2f",
            xticklabels=col_labels, yticklabels=row_labels, cbar_kws={'label': 'Mean Reward'})
ax.set_xlabel("Style")
ax.set_ylabel("Difficulty")
ax.set_title("Mean Reward by Difficulty & Style")

plt.tight_layout()
plt.savefig("plots/correlation_analysis.png", dpi=150, bbox_inches="tight")
print("Saved: plots/correlation_analysis.png")

# Additional: Joint distribution
fig2, ax2 = plt.subplots(figsize=(10, 8))
scatter = ax2.scatter(difficulties, styles, c=rewards, cmap="viridis", alpha=0.6, s=20)
cbar = plt.colorbar(scatter, ax=ax2)
cbar.set_label("Reward")
ax2.set_xlabel("Difficulty")
ax2.set_ylabel("Style")
ax2.set_title("Joint Distribution: Difficulty, Style, and Reward")
plt.savefig("plots/joint_distribution.png", dpi=150, bbox_inches="tight")
print("Saved: plots/joint_distribution.png")

print("\n" + "=" * 60)
print("INTERPRETATION")
print("=" * 60)
print(f"""
Correlation Strength Guide:
  |r| < 0.1  : Negligible
  |r| 0.1-0.3: Weak
  |r| 0.3-0.5: Moderate
  |r| 0.5-0.7: Strong
  |r| > 0.7  : Very strong

Key Findings:
  - Difficulty-Reward correlation: {r_diff:+.4f} ({'positive' if r_diff > 0 else 'negative'}, {'significant' if p_diff < 0.05 else 'not significant'})
  - Style-Reward correlation: {r_style:+.4f} ({'positive' if r_style > 0 else 'negative'}, {'significant' if p_style < 0.05 else 'not significant'})
  - Difficulty-Style correlation: {r_diff_style:+.4f} ({'positive' if r_diff_style > 0 else 'negative'}, {'significant' if p_diff_style < 0.05 else 'not significant'})
""")
