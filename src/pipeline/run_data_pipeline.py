"""Run the Checkpoint 2 data pipeline end to end."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def run_step(command: list[str]) -> None:
    """Run a pipeline step and fail fast on errors."""
    print("\n$", " ".join(command))
    subprocess.run(command, check=True)


def has_normalized_snapshot() -> bool:
    """Return whether a normalized raw snapshot is available for cleaning."""
    return any(Path("data/raw").glob("cian_spb_normalized_*.csv"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run CIAN ETL and feature pipeline.")
    parser.add_argument("--collect", action="store_true", help="Collect a new CIAN snapshot first.")
    parser.add_argument("--pages", type=int, default=10, help="Pages per room segment if --collect is used.")
    parser.add_argument("--timeout", type=int, default=20, help="Request timeout if --collect is used.")
    parser.add_argument("--skip-eda", action="store_true", help="Skip EDA figure generation.")
    parser.add_argument("--skip-baseline", action="store_true", help="Skip baseline metric generation.")
    parser.add_argument(
        "--with-routing",
        action="store_true",
        help="Use openrouteservice API to calculate walking route distance to metro.",
    )
    args = parser.parse_args()

    python = sys.executable

    if args.collect:
        run_step([
            python,
            "-m",
            "src.data.collect_cian_spb",
            "--pages",
            str(args.pages),
            "--timeout",
            str(args.timeout),
        ])

    clean_path = Path("data/processed/cian_spb_clean.csv")
    if args.collect or has_normalized_snapshot():
        run_step([python, "-m", "src.data.clean_cian"])
    elif clean_path.exists():
        print(
            f"\nNo normalized raw snapshot found; reusing existing clean dataset: {clean_path}"
        )
    else:
        raise FileNotFoundError(
            "No normalized raw snapshot or clean dataset found. "
            "Run with `--collect` or add data/processed/cian_spb_clean.csv."
        )
    run_step([python, "-m", "src.data.contract_cian", "data/processed/cian_spb_clean.csv"])
    geocoder_command = [python, "-m", "src.features.geocoder"]
    if args.with_routing:
        geocoder_command.append("--with-routing")
    run_step(geocoder_command)
    run_step([python, "-m", "src.features.build_features"])
    run_step([python, "-m", "src.data.sampling"])

    if not args.skip_eda:
        run_step([python, "-m", "src.data.make_cian_eda"])
    if not args.skip_baseline:
        run_step([python, "-m", "src.models.baseline_cian"])
    run_step([python, "-m", "src.pipeline.make_pipeline_report"])

    print("\nPipeline completed.")
    print("Main artifacts:")
    for path in [
        "data/processed/cian_spb_clean.csv",
        "data/processed/cian_spb_clean_geo.csv",
        "data/features/cian_spb_offline_features.csv",
        "data/processed/cian_spb_balanced_sample.csv",
        "data/processed/sampling_report.md",
        "data/processed/pipeline_report.md",
        "docs/feature_registry.md",
    ]:
        print(f"  - {Path(path)}")


if __name__ == "__main__":
    main()
