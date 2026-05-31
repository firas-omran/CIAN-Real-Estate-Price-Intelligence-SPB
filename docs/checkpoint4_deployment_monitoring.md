# Checkpoint 4. Deployment, Monitoring And Operations

**Project:** CIAN Real Estate Price Intelligence SPB  
**Goal:** deploy a working ML application and demonstrate monitoring, alerts,
runbook actions, and data drift reaction.

## 1. Deployment Strategy

### Serving Components

| Component | Tool | Purpose |
|---|---|---|
| Prediction API | FastAPI + Uvicorn | `/predict`, `/health`, Swagger demo |
| Web UI | Streamlit | student-facing demo and monitoring dashboard |
| Orchestration | Airflow DAG | scheduled data pipeline + experiments |
| Experiment tracking | MLflow | runs, metrics, params, model artifacts |
| Event streaming | Kafka | prediction events for operational analysis |
| Metrics | Prometheus | scrape API `/metrics` |
| Dashboards | Grafana | production metrics dashboard |
| Model artifact | `data/models/random_forest.pkl` | current best model from Checkpoint 3 |
| Feature/data artifacts | CSV + JSON | offline features, geocode cache, experiment metadata |

The current serving model is selected automatically from:

```text
data/experiments/checkpoint3_metrics.csv
```

For the latest fresh snapshot the best model is:

```text
random_forest
test_r2_per_sqm = 0.6820
test_mae_per_sqm = 72,103 RUB/m2
test_mape = 16.6%
```

### Deployment Choice

For MVP defense we use a simple two-service deployment:

1. **FastAPI** runs the model endpoint.
2. **Streamlit** provides the user interface and monitoring/drift demo.

Recommended hosting options:

| Option | Use |
|---|---|
| Local laptop demo | primary classroom defense, lowest risk |
| Render / Railway | public API + Streamlit if internet demo is required |
| Docker Compose | reproducible local or VM deployment |

The optional production-like stack is defined in:

```text
docker-compose.yml
```

It includes:

- FastAPI;
- Streamlit;
- MLflow;
- Kafka + Kafka UI;
- Prometheus;
- Grafana;
- Airflow.

### Local Run Commands

API:

```bash
arch -arm64 python -m uvicorn src.api.main:app --reload --port 8000
```

Web app:

```bash
arch -arm64 python -m streamlit run src/app/streamlit_app.py
```

Full MLOps stack:

```bash
docker compose up --build
```

Service URLs:

| Service | URL |
|---|---|
| Streamlit | http://localhost:8501 |
| FastAPI docs | http://localhost:8000/docs |
| FastAPI metrics | http://localhost:8000/metrics |
| MLflow | http://localhost:5001 |
| Kafka UI | http://localhost:8085 |
| Prometheus | http://localhost:9090 |
| Grafana | http://localhost:3000 (`admin` / `admin`) |
| Airflow | http://localhost:8080 (`admin` / `admin`) |

### Rollback Plan

Rollback unit:

- application code version;
- model artifact in `data/models`;
- experiment metadata in `data/experiments/checkpoint3_metadata.json`;
- feature snapshot hash.

Rollback steps:

1. Stop the current API/UI process.
2. Restore the previous accepted model artifact.
3. Restore previous `checkpoint3_metrics.csv` and metadata if model selection changed.
4. Start API and check `/health`.
5. Run one known prediction request from Swagger.
6. Keep the failed model/data snapshot for incident analysis.

Acceptance after rollback:

- `/health` returns `status=ok`;
- `/predict` returns a finite positive price;
- data contract passes;
- no system alert for latency/error rate.

## 2. Monitoring

Monitoring is implemented in:

```text
src/monitoring/checkpoint4.py
```

It is exposed in the Streamlit tab:

```text
Monitoring & Drift
```

Production metrics are also exposed by FastAPI at:

```text
/metrics
```

Prometheus scrapes these metrics and Grafana provisions a dashboard from:

```text
ops/grafana/dashboards/cian_api_dashboard.json
```

### 2.1 Data Monitoring

| Signal | Metric | Threshold |
|---|---|---:|
| Freshness | snapshot age from `collected_at` | > 7 days |
| Data quality | share of rows with `lat/lon` | < 95% |
| Missing target | missing `target_price_per_sqm` | > 0% |
| Numeric drift | PSI for area, rooms, distance, target | > 0.20 |
| Segment drift | max category share delta | > 10 pp |

Numeric drift uses Population Stability Index (PSI). Categorical drift uses
maximum absolute share change across categories.

### 2.2 Model Monitoring

| Signal | Metric | Threshold |
|---|---|---:|
| Prediction distribution | median and p95 predicted price | tracked |
| Luxury/outlier pressure | share of predictions > 100M RUB | > 5% |
| Active model | selected model name | must match expected artifact |

The demo calculates batch predictions on the current monitoring snapshot and
tracks the prediction distribution.

### 2.3 System Monitoring

| Signal | Metric | Threshold |
|---|---|---:|
| Health | `/health` status | not ok |
| Latency | p95 response time | > 500 ms |
| Errors | API error rate | > 5% |

For the classroom demo, system metrics are simulated in the degradation
scenario. In production they would be collected from API logs or Prometheus.

### 2.4 MLflow Experiment Tracking

`src/models/experiments.py` logs every model run to MLflow when available:

- dataset hash;
- selected features;
- model parameters;
- CV/validation/test metrics;
- model artifact.

If `MLFLOW_TRACKING_URI` is set, the script uses it. Otherwise it logs locally:

```text
data/mlruns
```

### 2.5 Kafka Prediction Events

FastAPI publishes one event per prediction when `KAFKA_BOOTSTRAP_SERVERS` is set.
Topic:

```text
cian.predictions
```

The event contains model name, price prediction, request segment, and latency.
Kafka is not required for serving; it is an operational event stream for
audit, dashboards, and future online monitoring.

### 2.6 Airflow Orchestration

The Airflow DAG is:

```text
dags/cian_ml_pipeline_dag.py
```

It runs:

1. data collection and pipeline;
2. data contract validation;
3. modeling experiments with MLflow logging;
4. retraining gate dry-run.

## 3. Alerts

| Alert | Level | Condition | Cause | Action |
|---|---|---|---|---|
| `snapshot_freshness` | data | age > 7 days | stale CIAN snapshot | run collection pipeline, validate contract, rebuild features |
| `geocode_coverage_drop` | data | geocode coverage < 95% | Nominatim/cache/SSL issue or incomplete addresses | check geocoder logs/certificates/cache, rerun geocoding |
| `numeric_data_drift` | data | max PSI > 0.20 | area, price, or location distribution shifted | inspect drift table, retrain if shift is real |
| `segment_drift` | data | category share delta > 10 pp | district/room/seller mix changed | check parser query, rebalance or retrain |
| `prediction_tail_growth` | model | predictions > 100M RUB exceed 5% | many luxury/outlier-like requests | inspect high-price requests, cap demo ranges, retrain with better coverage |
| `api_latency_p95` | system | p95 latency > 500 ms | cold start, host overload, slow geocoding path | restart, disable live geocoding, rollback if needed |
| `api_error_rate` | system | error rate > 5% | bad release, missing model, invalid inputs | rollback, inspect stack traces, check artifacts |

## 4. Runbook

### Alert: `snapshot_freshness`

1. Confirm latest `collected_at`.
2. Run:

```bash
python -m src.pipeline.run_data_pipeline --collect --pages 20 --timeout 20 --with-routing
```

3. Run data contract.
4. Run experiments and compare metrics.

### Alert: `geocode_coverage_drop`

1. Check whether SSL certificates are installed on macOS.
2. Check Nominatim availability and geocode cache.
3. Rerun:

```bash
python -m src.features.geocoder --with-routing
```

4. If live geocoding is unstable, use district fallback for demo and mark
geo precision as degraded.

### Alert: `numeric_data_drift`

1. Open the drift table in Streamlit.
2. Identify features with PSI > 0.20.
3. Check whether the shift is real market movement or parser bias.
4. If real: retrain and compare to previous accepted model.
5. If parser issue: fix collection query and recollect data.

### Alert: `segment_drift`

1. Inspect changed categories: district, rooms, seller type, geo precision.
2. Verify that CIAN collection still uses the intended room segments.
3. Rebalance sampling if one segment dominates.
4. Retrain if the new segment mix is expected.

### Alert: `prediction_tail_growth`

1. Inspect high-price predictions and input rows.
2. Check for luxury outliers or invalid area/floor values.
3. Tighten UI/API validation if needed.
4. Retrain with fresh data if the market mix changed.

### Alert: `api_latency_p95`

1. Check whether the model loads once and is cached.
2. Disable live geocoding on the hot prediction path.
3. Restart API.
4. Roll back if latency started after a release.

### Alert: `api_error_rate`

1. Check `/health`.
2. Check model artifact exists in `data/models`.
3. Check stack traces.
4. Roll back code/model artifacts.
5. Re-run one Swagger `/predict` request.

## 5. Demonstration Plan

### Working Application

1. Open Streamlit.
2. Enter apartment parameters:

```text
rooms = 2
area = 58
floor = 5
floors_count = 12
district = Василеостровский
metro = Приморская
seller = real_estate_agent
```

3. Press `Оценить стоимость`.
4. Show predicted price, price per square meter, active model, geo precision,
distance to center, and distance to metro.

### Data Drift / Degradation

1. Open the `Monitoring & Drift` tab.
2. Select `Normal snapshot`.
3. Show that alerts are inactive or minimal.
4. Select `Degradation / drift demo`.
5. Explain the simulated degradation:

- larger apartment areas;
- higher price-per-square-meter segment;
- shift toward central/premium districts;
- lower geocode coverage;
- higher API latency and error rate.

6. Show fired alerts:

- numeric data drift;
- segment drift;
- geocode coverage drop;
- stale snapshot;
- model prediction tail growth;
- system latency/error alerts.

7. Show the runbook actions for each fired alert.

This satisfies the supervisor requirement:

```text
Надо показать дрейф данных
```
