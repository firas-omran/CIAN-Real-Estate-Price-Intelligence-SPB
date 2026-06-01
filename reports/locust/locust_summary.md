# Locust Load Test Summary

SLO:

```text
p95 latency <= 500 ms
error rate <= 5.00%
```

| Scenario | Users | RPS | p95 latency, ms | p99 latency, ms | Error rate | Result |
|---|---:|---:|---:|---:|---:|---|
| smoke | 1 | 6.39 | 39.00 | 47.00 | 0.00% | PASSED |
| demo | 10 | 58.07 | 90.00 | 120.00 | 0.00% | PASSED |
| stress | 30 | 61.26 | 560.00 | 630.00 | 0.00% | FAILED |

Conclusion: the service passed the configured SLO up to **10 concurrent users** in this run.

HTML reports:

- `reports/locust/smoke/report.html`
- `reports/locust/demo/report.html`
- `reports/locust/stress/report.html`
