"""Airflow DAG for the CIAN price-intelligence data and model pipeline."""

from __future__ import annotations

from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.bash import BashOperator

PROJECT_DIR = "/app"

default_args = {
    "owner": "cian-mlops",
    "depends_on_past": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

with DAG(
    dag_id="cian_spb_ml_pipeline",
    description="Collect CIAN data, validate contract, build features, train models, and check retraining triggers.",
    default_args=default_args,
    start_date=datetime(2026, 1, 1),
    schedule="@weekly",
    catchup=False,
    tags=["cian", "mlops", "checkpoint4"],
) as dag:
    run_data_pipeline = BashOperator(
        task_id="run_data_pipeline",
        bash_command=(
            f"cd {PROJECT_DIR} && "
            "python -m src.pipeline.run_data_pipeline --collect --pages 20 --timeout 20 --with-routing"
        ),
    )

    validate_contract = BashOperator(
        task_id="validate_data_contract",
        bash_command=(
            f"cd {PROJECT_DIR} && "
            "python -m src.data.contract_cian data/processed/cian_spb_clean_geo.csv"
        ),
    )

    run_experiments = BashOperator(
        task_id="run_model_experiments",
        bash_command=f"cd {PROJECT_DIR} && python -m src.models.experiments",
    )

    retraining_gate = BashOperator(
        task_id="retraining_gate_dry_run",
        bash_command=f"cd {PROJECT_DIR} && python -m src.pipeline.auto_retrain --dry-run",
    )

    run_data_pipeline >> validate_contract >> run_experiments >> retraining_gate
