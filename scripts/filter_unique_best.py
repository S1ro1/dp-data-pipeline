#!/usr/bin/env python3
"""
Keep only the best entry per module_name based on reward score.

Usage:
    uv run python scripts/filter_unique_best.py              # Full run with upload
    uv run python scripts/filter_unique_best.py --no-upload  # Skip upload
"""

import argparse

import config


def calculate_score(entry: dict) -> float:
    """Calculate score for entry. Returns reward value."""
    return entry["reward"]


def main():
    parser = argparse.ArgumentParser(description="Deduplicate by module, keep best")
    parser.add_argument(
        "--no-upload",
        action="store_true",
        help="Skip uploading to HuggingFace",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("Unique Best Filter")
    print("=" * 60)
    config.print_config()

    input_path = config.get_filtered_path()
    output_path = config.get_unique_path()

    # Load all entries
    entries = config.load_jsonl(input_path)
    print(f"\nLoaded {len(entries):,} entries from {input_path}")

    # Group by module_name and keep best
    best_by_module: dict[str, tuple[dict, float]] = {}

    for entry in entries:
        module_name = entry["info"]["module_name"]
        score = calculate_score(entry)

        if module_name not in best_by_module or score > best_by_module[module_name][1]:
            best_by_module[module_name] = (entry, score)

    # Extract best entries
    best_entries = [entry for entry, _ in best_by_module.values()]

    # Sort by score descending for nice output
    best_entries.sort(key=calculate_score, reverse=True)

    # Save output
    config.save_jsonl(best_entries, output_path)

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
        print("\nScore statistics (valid only):")
        print(f"  Min score:   {min(valid_scores):.2f}")
        print(f"  Max score:   {max(valid_scores):.2f}")
        print(f"  Avg score:   {sum(valid_scores) / len(valid_scores):.2f}")

    # Show top 10
    print("\nTop 10 entries by score:")
    for i, entry in enumerate(best_entries[:10], 1):
        score = calculate_score(entry)
        difficulty = entry.get("difficulty", "N/A")
        print(
            f"  {i:2d}. {entry.get('module_name', 'Unknown'):40s} score={score:.2f} d={difficulty}"
        )

    # Upload to HuggingFace
    if not args.no_upload:
        config.hf_login()
        config.upload_with_splits(best_entries, config.get_unique_repo())

    print("=" * 60)


if __name__ == "__main__":
    main()
