"""Sampling utilities for CIAN data."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


DEFAULT_INPUT = Path("data/processed/cian_spb_clean.csv")
DEFAULT_OUTPUT = Path("data/processed/cian_spb_balanced_sample.csv")
DEFAULT_REPORT = Path("data/processed/sampling_report.md")


def balanced_sample_by_rooms(df: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    """Downsample room segments to the size of the smallest segment."""
    counts = df["rooms_count"].value_counts()
    min_count = int(counts.min())
    sampled_parts = [
        part.sample(n=min_count, random_state=seed)
        for _, part in df.groupby("rooms_count")
    ]
    sampled = pd.concat(sampled_parts, ignore_index=True)
    sampled = sampled.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return sampled


def write_report(original: pd.DataFrame, sampled: pd.DataFrame, report_path: Path) -> None:
    """Write sampling summary for Checkpoint 2."""
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report = f"""# Sampling Report

Approach: stratified downsampling by `rooms_count`.

Reason: 1-room listings dominate the current CIAN snapshot. Balancing by room
segment prevents baseline/model experiments from being overly optimized for the
largest segment only.

Original rows: {len(original)}

Original room distribution:

{original["rooms_count"].value_counts().sort_index().to_string()}

Balanced rows: {len(sampled)}

Balanced room distribution:

{sampled["rooms_count"].value_counts().sort_index().to_string()}
"""
    report_path.write_text(report, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a balanced CIAN sample.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    sampled = balanced_sample_by_rooms(df, seed=args.seed)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    sampled.to_csv(args.output, index=False)
    write_report(df, sampled, args.report)

    print(f"Input: {args.input} -> {df.shape}")
    print(f"Balanced sample: {args.output} -> {sampled.shape}")
    print(sampled["rooms_count"].value_counts().sort_index().to_string())


if __name__ == "__main__":
    main()
