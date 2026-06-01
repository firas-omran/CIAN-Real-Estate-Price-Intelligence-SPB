# CIAN Load Test Report

## Run

| Metric | Value |
|---|---:|
| Serving URL | `http://api:8000` |
| Requests total | 100 |
| Concurrency | 10 |
| Success count | 100 |
| Error count | 0 |
| Error rate | 0.0000 |
| Throughput, RPS | 56.69 |
| Latency mean, ms | 171.44 |
| Latency p50, ms | 172.93 |
| Latency p95, ms | 237.78 |
| Latency p99, ms | 263.58 |
| Latency max, ms | 266.04 |
| SLO status | **PASSED** |

## SLO

| SLO | Threshold | Actual |
|---|---:|---:|
| p95 latency | 500 ms | 237.78 ms |
| error rate | 5.00% | 0.00% |

## Conclusion

The prediction service processed 100 / 100 requests successfully.
For the demo workload, the result is **PASSED** against the configured latency and error-rate SLO.
