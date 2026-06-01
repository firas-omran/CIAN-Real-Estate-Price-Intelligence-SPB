"""Concurrent load test for the CIAN FastAPI prediction endpoint."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import requests

DEFAULT_PAYLOADS = [
    {
        "rooms_count": 2,
        "total_meters": 58,
        "floor": 5,
        "floors_count": 12,
        "district": "Василеостровский",
        "underground": "Приморская",
        "author_type": "real_estate_agent",
        "street": "",
        "house_number": "",
    },
    {
        "rooms_count": 0,
        "total_meters": 32,
        "floor": 8,
        "floors_count": 16,
        "district": "Адмиралтейский",
        "underground": "Садовая",
        "author_type": "developer",
        "street": "",
        "house_number": "",
    },
    {
        "rooms_count": 3,
        "total_meters": 82,
        "floor": 4,
        "floors_count": 9,
        "district": "Петроградский",
        "underground": "Петроградская",
        "author_type": "realtor",
        "street": "",
        "house_number": "",
    },
    {
        "rooms_count": 1,
        "total_meters": 41,
        "floor": 2,
        "floors_count": 5,
        "district": "Калининский",
        "underground": "Площадь Мужества",
        "author_type": "homeowner",
        "street": "",
        "house_number": "",
    },
]


@dataclass
class RequestResult:
    ok: bool
    status_code: int | None
    latency_ms: float
    error: str | None = None


def percentile(values: list[float], q: float) -> float:
    """Return nearest-rank percentile."""
    if not values:
        return 0.0
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, round((len(ordered) - 1) * q)))
    return float(ordered[index])


def run_one(serving_url: str, payload: dict[str, Any], timeout: float) -> RequestResult:
    started = time.perf_counter()
    try:
        response = requests.post(f"{serving_url.rstrip('/')}/predict", json=payload, timeout=timeout)
        latency_ms = (time.perf_counter() - started) * 1000
        if response.status_code != 200:
            return RequestResult(False, response.status_code, latency_ms, response.text[:300])
        body = response.json()
        ok = bool(body.get("predicted_price", 0) > 0 and body.get("model_name"))
        return RequestResult(ok, response.status_code, latency_ms, None if ok else "invalid prediction body")
    except requests.RequestException as exc:
        latency_ms = (time.perf_counter() - started) * 1000
        return RequestResult(False, None, latency_ms, str(exc))


def summarize(results: list[RequestResult], started: float, finished: float, args: argparse.Namespace) -> dict[str, Any]:
    latencies = [item.latency_ms for item in results]
    successes = [item for item in results if item.ok]
    errors = [item for item in results if not item.ok]
    duration_seconds = max(finished - started, 1e-9)
    summary = {
        "serving_url": args.serving_url,
        "requests_total": args.requests,
        "concurrency": args.concurrency,
        "success_count": len(successes),
        "error_count": len(errors),
        "error_rate": len(errors) / max(len(results), 1),
        "duration_seconds": duration_seconds,
        "throughput_rps": len(results) / duration_seconds,
        "latency_mean_ms": statistics.mean(latencies) if latencies else 0.0,
        "latency_p50_ms": percentile(latencies, 0.50),
        "latency_p95_ms": percentile(latencies, 0.95),
        "latency_p99_ms": percentile(latencies, 0.99),
        "latency_max_ms": max(latencies) if latencies else 0.0,
        "slo": {
            "p95_latency_ms_threshold": args.p95_threshold_ms,
            "error_rate_threshold": args.error_rate_threshold,
            "passed": False,
        },
        "errors_sample": [asdict(item) for item in errors[:5]],
    }
    summary["slo"]["passed"] = (
        summary["latency_p95_ms"] <= args.p95_threshold_ms
        and summary["error_rate"] <= args.error_rate_threshold
    )
    return summary


def write_markdown_report(summary: dict[str, Any], path: Path) -> None:
    status = "PASSED" if summary["slo"]["passed"] else "FAILED"
    report = f"""# CIAN Load Test Report

## Run

| Metric | Value |
|---|---:|
| Serving URL | `{summary["serving_url"]}` |
| Requests total | {summary["requests_total"]} |
| Concurrency | {summary["concurrency"]} |
| Success count | {summary["success_count"]} |
| Error count | {summary["error_count"]} |
| Error rate | {summary["error_rate"]:.4f} |
| Throughput, RPS | {summary["throughput_rps"]:.2f} |
| Latency mean, ms | {summary["latency_mean_ms"]:.2f} |
| Latency p50, ms | {summary["latency_p50_ms"]:.2f} |
| Latency p95, ms | {summary["latency_p95_ms"]:.2f} |
| Latency p99, ms | {summary["latency_p99_ms"]:.2f} |
| Latency max, ms | {summary["latency_max_ms"]:.2f} |
| SLO status | **{status}** |

## SLO

| SLO | Threshold | Actual |
|---|---:|---:|
| p95 latency | {summary["slo"]["p95_latency_ms_threshold"]:.0f} ms | {summary["latency_p95_ms"]:.2f} ms |
| error rate | {summary["slo"]["error_rate_threshold"]:.2%} | {summary["error_rate"]:.2%} |

## Conclusion

The prediction service processed {summary["success_count"]} / {summary["requests_total"]} requests successfully.
For the demo workload, the result is **{status}** against the configured latency and error-rate SLO.
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a concurrent load test against CIAN /predict.")
    parser.add_argument("--serving-url", default="http://localhost:8000")
    parser.add_argument("--requests", type=int, default=100)
    parser.add_argument("--concurrency", type=int, default=10)
    parser.add_argument("--timeout", type=float, default=10.0)
    parser.add_argument("--output", type=Path, default=Path("reports/load_test_results.json"))
    parser.add_argument("--markdown-report", type=Path, default=Path("reports/load_test_report.md"))
    parser.add_argument("--p95-threshold-ms", type=float, default=500.0)
    parser.add_argument("--error-rate-threshold", type=float, default=0.05)
    args = parser.parse_args()

    started = time.perf_counter()
    results: list[RequestResult] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        futures = [
            executor.submit(run_one, args.serving_url, DEFAULT_PAYLOADS[i % len(DEFAULT_PAYLOADS)], args.timeout)
            for i in range(args.requests)
        ]
        for future in concurrent.futures.as_completed(futures):
            results.append(future.result())
    finished = time.perf_counter()

    summary = summarize(results, started, finished, args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_markdown_report(summary, args.markdown_report)

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    raise SystemExit(0 if summary["slo"]["passed"] else 1)


if __name__ == "__main__":
    main()
