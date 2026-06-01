"""Run smoke, demo, and stress Locust load-test scenarios."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class Scenario:
    name: str
    users: int
    spawn_rate: int
    run_time: str


SCENARIOS = [
    Scenario("smoke", users=1, spawn_rate=1, run_time="1m"),
    Scenario("demo", users=10, spawn_rate=5, run_time="5m"),
    Scenario("stress", users=30, spawn_rate=10, run_time="5m"),
]


def read_stats(csv_prefix: Path) -> dict[str, float | int | str]:
    stats_path = csv_prefix.with_name(csv_prefix.name + "_stats.csv")
    with stats_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    aggregate = next(row for row in rows if row["Name"] == "Aggregated")
    requests = int(float(aggregate["Request Count"]))
    failures = int(float(aggregate["Failure Count"]))
    error_rate = failures / max(requests, 1)
    return {
        "requests_total": requests,
        "failure_count": failures,
        "error_rate": error_rate,
        "throughput_rps": float(aggregate["Requests/s"]),
        "latency_mean_ms": float(aggregate["Average Response Time"]),
        "latency_p50_ms": float(aggregate["50%"]),
        "latency_p95_ms": float(aggregate["95%"]),
        "latency_p99_ms": float(aggregate["99%"]),
        "latency_max_ms": float(aggregate["Max Response Time"]),
    }


def run_scenario(args: argparse.Namespace, scenario: Scenario) -> dict:
    output_dir = args.output_dir / scenario.name
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_prefix = output_dir / "locust"
    html_path = output_dir / "report.html"

    command = [
        sys.executable,
        "-m",
        "locust",
        "-f",
        str(args.locustfile),
        "--host",
        args.host,
        "--users",
        str(scenario.users),
        "--spawn-rate",
        str(scenario.spawn_rate),
        "--run-time",
        scenario.run_time,
        "--headless",
        "--html",
        str(html_path),
        "--csv",
        str(csv_prefix),
        "--only-summary",
    ]
    subprocess.run(command, check=True)
    metrics = read_stats(csv_prefix)
    passed = (
        metrics["latency_p95_ms"] <= args.p95_threshold_ms
        and metrics["error_rate"] <= args.error_rate_threshold
    )
    return {
        **asdict(scenario),
        **metrics,
        "html_report": str(html_path),
        "slo_passed": passed,
    }


def write_summary(results: list[dict], args: argparse.Namespace) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "locust_summary.json"
    md_path = args.output_dir / "locust_summary.md"
    json_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

    rows = "\n".join(
        "| {name} | {users} | {throughput_rps:.2f} | {latency_p95_ms:.2f} | {latency_p99_ms:.2f} | {error_rate:.2%} | {result} |".format(
            **item,
            result="PASSED" if item["slo_passed"] else "FAILED",
        )
        for item in results
    )
    max_passed = [item for item in results if item["slo_passed"]]
    capacity = max((item["users"] for item in max_passed), default=0)
    report = f"""# Locust Load Test Summary

SLO:

```text
p95 latency <= {args.p95_threshold_ms:.0f} ms
error rate <= {args.error_rate_threshold:.2%}
```

| Scenario | Users | RPS | p95 latency, ms | p99 latency, ms | Error rate | Result |
|---|---:|---:|---:|---:|---:|---|
{rows}

Conclusion: the service passed the configured SLO up to **{capacity} concurrent users** in this run.

HTML reports:

{chr(10).join(f'- `{item["html_report"]}`' for item in results)}
"""
    md_path.write_text(report, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run all Locust load-test scenarios.")
    parser.add_argument("--host", default="http://localhost:8000")
    parser.add_argument("--locustfile", type=Path, default=Path("load_tests/locustfile.py"))
    parser.add_argument("--output-dir", type=Path, default=Path("reports/locust"))
    parser.add_argument("--p95-threshold-ms", type=float, default=500.0)
    parser.add_argument("--error-rate-threshold", type=float, default=0.05)
    args = parser.parse_args()

    results = [run_scenario(args, scenario) for scenario in SCENARIOS]
    write_summary(results, args)
    if not all(item["slo_passed"] for item in results[:2]):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
