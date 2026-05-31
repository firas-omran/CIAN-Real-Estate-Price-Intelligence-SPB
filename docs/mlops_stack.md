# Optional MLOps Stack

This project can be demonstrated as a lightweight production-like ML system.

## Components

| Tool | Role |
|---|---|
| Airflow | weekly orchestration of data collection, validation, training, retraining gate |
| MLflow | experiment tracking, params, metrics, model artifacts |
| Kafka | prediction event stream |
| Prometheus | scrape FastAPI `/metrics` |
| Grafana | dashboard for prediction traffic, latency, and price distribution |

## Run

```bash
docker compose up --build
```

Open:

```text
Streamlit:  http://localhost:8501
FastAPI:    http://localhost:8000/docs
MLflow:     http://localhost:5001
Kafka UI:   http://localhost:8085
Prometheus: http://localhost:9090
Grafana:    http://localhost:3000  admin/admin
Airflow:    http://localhost:8080  admin/admin
```

## Airflow

DAG:

```text
dags/cian_ml_pipeline_dag.py
```

Pipeline:

```text
run_data_pipeline -> validate_data_contract -> run_model_experiments -> retraining_gate_dry_run
```

## MLflow

Experiment name:

```text
cian_spb_price_intelligence
```

Run manually:

```bash
MLFLOW_TRACKING_URI=http://localhost:5001 python -m src.models.experiments
```

## Kafka

Topic:

```text
cian.predictions
```

FastAPI publishes prediction events when:

```text
KAFKA_BOOTSTRAP_SERVERS
```

is set.

## Prometheus And Grafana

FastAPI exposes:

```text
http://localhost:8000/metrics
```

Dashboard:

```text
ops/grafana/dashboards/cian_api_dashboard.json
```
