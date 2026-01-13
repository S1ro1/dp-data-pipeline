#!/usr/bin/env python3
"""
Keep only the best entry per module_name based on total_score (reward * difficulty * style).
"""

import json
from pathlib import Path

INPUT_FILE = "outputs/filtered_dataset.jsonl"
OUTPUT_FILE = "outputs/filtered_dataset-filtered.jsonl"


def calculate_score(entry: dict) -> float:
    """Calculate total_score = reward * difficulty * style. Returns 0 if any value is None."""
    reward = entry.get("reward", 0) or 0
    difficulty = entry.get("difficulty")
    style = entry.get("style")

    if difficulty is None or style is None:
        return 0.0

    return reward * difficulty * style


def main():
    input_path = Path(INPUT_FILE)
    output_path = Path(OUTPUT_FILE)

    # Load all entries
    entries = []
    with input_path.open("r") as f:
        for line in f:
            if line.strip():
                entries.append(json.loads(line))

    print(f"Loaded {len(entries):,} entries from {INPUT_FILE}")

    # Group by module_name and keep best
    best_by_module: dict[str, tuple[dict, float]] = {}

    for entry in entries:
        module_name = entry.get("module_name", "Unknown")
        score = calculate_score(entry)

        if module_name not in best_by_module or score > best_by_module[module_name][1]:
            best_by_module[module_name] = (entry, score)

    # Extract best entries
    best_entries = [entry for entry, _ in best_by_module.values()]

    # Sort by score descending for nice output
    best_entries.sort(key=calculate_score, reverse=True)

    # Write output
    with output_path.open("w") as f:
        for entry in best_entries:
            f.write(json.dumps(entry) + "\n")

    # Statistics
    scores = [calculate_score(e) for e in best_entries]
    valid_scores = [s for s in scores if s > 0]

    print(f"\n{'=' * 60}")
    print("FILTERING RESULTS")
    print(f"{'=' * 60}")
    print(f"Original entries:          {len(entries):,}")
    print(f"Unique modules (kept):     {len(best_entries):,}")
    print(f"Entries removed:           {len(entries) - len(best_entries):,}")
    print(
        f"Reduction:                 {(1 - len(best_entries) / len(entries)) * 100:.1f}%"
    )
    print(f"\nEntries with valid score:  {len(valid_scores):,}")
    print(f"Entries with score = 0:    {len(scores) - len(valid_scores):,}")

    if valid_scores:
        print(f"\nScore statistics (valid only):")
        print(f"  Min score:   {min(valid_scores):.2f}")
        print(f"  Max score:   {max(valid_scores):.2f}")
        print(f"  Avg score:   {sum(valid_scores) / len(valid_scores):.2f}")

    print(f"\nOutput saved to: {OUTPUT_FILE}")
    print(f"{'=' * 60}")

    # Show top 10
    print("\nTop 10 entries by total_score:")
    for i, entry in enumerate(best_entries[:10], 1):
        score = calculate_score(entry)
        print(
            f"  {i:2d}. {entry['module_name']:40s} score={score:.2f} "
            f"(r={entry['reward']:.2f}, d={entry['difficulty']}, s={entry['style']})"
        )


if __name__ == "__main__":
    main()
