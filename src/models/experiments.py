"""Reproducible modeling experiments for Checkpoint 3.

Runs multiple model classes on the same fixed data snapshot, stores model
parameters, data hash, feature list, metrics, and a compact Markdown report.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import KFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

DEFAULT_INPUT = Path("data/features/cian_spb_offline_features.csv")
DEFAULT_OUTPUT_DIR = Path("data/experiments")
DEFAULT_MODEL_DIR = Path("data/models")
RANDOM_STATE = 42
N_SPLITS = 5
TARGET = "log_target_price_per_sqm"
BASELINE_R2_PER_SQM = 0.313709

LEAKAGE_COLUMNS = {
    "listing_id",
    "url",
    "source",
    "collected_at",
    "location_query",
    "price",
    "log_price",
    "target_price_per_sqm",
    "log_target_price_per_sqm",
    "residential_complex",
}
AGGREGATE_PATTERNS = (
    "_ads_count",
    "_median_price",
    "_median_price_per_sqm",
    "_p25_price_per_sqm",
    "_p75_price_per_sqm",
)

NUMERIC_FEATURES = [
    "rooms_count",
    "total_meters",
    "log_total_meters",
    "floor",
    "floors_count",
    "floor_ratio",
    "is_first_floor",
    "is_last_floor",
    "lat",
    "lon",
    "distance_to_center_km",
    "distance_to_metro_route_km",
    "duration_to_metro_route_min",
    "metro_known",
]

CATEGORICAL_FEATURES = [
    "author_type",
    "room_segment",
    "district",
    "underground",
    "geo_precision",
    "metro_route_provider",
]


def file_sha256(path: Path) -> str:
    """Return SHA-256 hash for the exact dataset artifact used."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def is_aggregate_column(column: str) -> bool:
    """Return whether a column is a target-derived market aggregate."""
    return any(pattern in column for pattern in AGGREGATE_PATTERNS)


def select_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, list[str], list[str]]:
    """Select non-leakage model features and split numeric/categorical lists."""
    feature_columns = [
        column
        for column in df.columns
        if column not in LEAKAGE_COLUMNS and not is_aggregate_column(column)
    ]
    numeric = [column for column in NUMERIC_FEATURES if column in feature_columns]
    categorical = [column for column in CATEGORICAL_FEATURES if column in feature_columns]
    selected = numeric + categorical
    return df[selected].copy(), df[TARGET].copy(), numeric, categorical


def split_data(X: pd.DataFrame, y: pd.Series) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """Use one reproducible 60/20/20 split for all experiments."""
    stratify = X["rooms_count"].astype(int)
    x_train, x_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=0.4,
        random_state=RANDOM_STATE,
        stratify=stratify,
    )
    x_val, x_test, y_val, y_test = train_test_split(
        x_temp,
        y_temp,
        test_size=0.5,
        random_state=RANDOM_STATE,
        stratify=x_temp["rooms_count"].astype(int),
    )
    return x_train, x_val, x_test, y_train, y_val, y_test


def make_preprocessor(numeric_features: list[str], categorical_features: list[str], scale_numeric: bool) -> ColumnTransformer:
    """Build preprocessing with fixed feature order."""
    numeric_steps: list[tuple[str, Any]] = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))

    numeric_pipe = Pipeline(numeric_steps)
    categorical_pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="constant", fill_value="unknown")),
            ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False, max_categories=60)),
        ]
    )

    return ColumnTransformer(
        [
            ("num", numeric_pipe, numeric_features),
            ("cat", categorical_pipe, categorical_features),
        ],
        remainder="drop",
    )


def build_model_specs(numeric_features: list[str], categorical_features: list[str]) -> dict[str, Pipeline]:
    """Return at least two model classes: linear and ensemble/boosting."""
    return {
        "ridge_linear": Pipeline(
            [
                ("prep", make_preprocessor(numeric_features, categorical_features, scale_numeric=True)),
                ("model", Ridge(alpha=1.0, random_state=RANDOM_STATE)),
            ]
        ),
        "random_forest": Pipeline(
            [
                ("prep", make_preprocessor(numeric_features, categorical_features, scale_numeric=False)),
                (
                    "model",
                    RandomForestRegressor(
                        n_estimators=400,
                        max_depth=14,
                        min_samples_leaf=3,
                        random_state=RANDOM_STATE,
                        n_jobs=1,
                    ),
                ),
            ]
        ),
        "gradient_boosting": Pipeline(
            [
                ("prep", make_preprocessor(numeric_features, categorical_features, scale_numeric=False)),
                (
                    "model",
                    GradientBoostingRegressor(
                        n_estimators=350,
                        learning_rate=0.04,
                        max_depth=3,
                        min_samples_leaf=4,
                        subsample=0.9,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
    }


def per_sqm_metrics(y_true_log: pd.Series, y_pred_log: np.ndarray) -> dict[str, float]:
    """Compute metrics on the original RUB/m2 scale."""
    actual = np.expm1(y_true_log.to_numpy())
    predicted = np.expm1(y_pred_log)
    mask = actual > 0
    actual = actual[mask]
    predicted = predicted[mask]
    ape = np.abs(predicted - actual) / actual
    return {
        "r2_per_sqm": float(r2_score(actual, predicted)),
        "mae_per_sqm": float(mean_absolute_error(actual, predicted)),
        "mape_percent": float(ape.mean() * 100),
    }


def collect_model_params(model: Pipeline) -> dict[str, Any]:
    """Return JSON-friendly estimator parameters."""
    estimator = model.named_steps["model"]
    params = estimator.get_params()
    return {
        key: value
        for key, value in params.items()
        if isinstance(value, (str, int, float, bool, type(None)))
    }


def run_experiments(input_path: Path, output_dir: Path, model_dir: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Run all experiments and persist metrics, metadata, and models."""
    df = pd.read_csv(input_path)
    X, y, numeric_features, categorical_features = select_features(df)
    x_train, x_val, x_test, y_train, y_val, y_test = split_data(X, y)
    model_specs = build_model_specs(numeric_features, categorical_features)

    output_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = [
        {
            "model": "B2_non_ml_baseline",
            "model_class": "group_median_baseline",
            "cv_r2_log_mean": np.nan,
            "cv_r2_log_std": np.nan,
            "val_r2_per_sqm": np.nan,
            "val_mae_per_sqm": np.nan,
            "val_mape_percent": np.nan,
            "test_r2_per_sqm": BASELINE_R2_PER_SQM,
            "test_mae_per_sqm": np.nan,
            "test_mape_percent": np.nan,
            "improvement_vs_baseline_r2": 0.0,
        }
    ]
    model_metadata: dict[str, Any] = {}

    cv = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    for name, model in model_specs.items():
        print(f"\n=== {name} ===")
        cv_scores = cross_validate(
            model,
            x_train,
            y_train,
            cv=cv,
            scoring={"r2": "r2", "neg_mae": "neg_mean_absolute_error"},
            n_jobs=1,
        )
        model.fit(x_train, y_train)
        val_pred = model.predict(x_val)
        test_pred = model.predict(x_test)
        val_metrics = per_sqm_metrics(y_val, val_pred)
        test_metrics = per_sqm_metrics(y_test, test_pred)

        estimator = model.named_steps["model"]
        row = {
            "model": name,
            "model_class": estimator.__class__.__name__,
            "cv_r2_log_mean": float(cv_scores["test_r2"].mean()),
            "cv_r2_log_std": float(cv_scores["test_r2"].std()),
            "val_r2_per_sqm": val_metrics["r2_per_sqm"],
            "val_mae_per_sqm": val_metrics["mae_per_sqm"],
            "val_mape_percent": val_metrics["mape_percent"],
            "test_r2_per_sqm": test_metrics["r2_per_sqm"],
            "test_mae_per_sqm": test_metrics["mae_per_sqm"],
            "test_mape_percent": test_metrics["mape_percent"],
            "improvement_vs_baseline_r2": test_metrics["r2_per_sqm"] - BASELINE_R2_PER_SQM,
        }
        rows.append(row)
        model_metadata[name] = {
            "model_class": estimator.__class__.__name__,
            "params": collect_model_params(model),
            "model_path": str(model_dir / f"{name}.pkl"),
        }
        joblib.dump(model, model_dir / f"{name}.pkl")
        print(
            f"test R2={row['test_r2_per_sqm']:.4f} "
            f"MAE={row['test_mae_per_sqm']:,.0f} "
            f"MAPE={row['test_mape_percent']:.1f}%"
        )

    metrics = pd.DataFrame(rows).sort_values("test_r2_per_sqm", ascending=False)
    metadata = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "input_path": str(input_path),
        "dataset_sha256": file_sha256(input_path),
        "dataset_shape": {"rows": int(df.shape[0]), "columns": int(df.shape[1])},
        "selected_feature_count": len(X.columns),
        "selected_features": list(X.columns),
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "target": TARGET,
        "split": {
            "strategy": "train/val/test",
            "train_rows": int(x_train.shape[0]),
            "val_rows": int(x_val.shape[0]),
            "test_rows": int(x_test.shape[0]),
            "ratio": "60/20/20",
            "stratify": "rooms_count",
            "random_state": RANDOM_STATE,
        },
        "baseline": {
            "name": "B2_non_ml_baseline",
            "definition": "median price_per_sqm by district + rooms_count",
            "r2_per_sqm": BASELINE_R2_PER_SQM,
        },
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "sklearn": sklearn.__version__,
        },
        "models": model_metadata,
    }
    return metrics, metadata


def write_report(metrics: pd.DataFrame, metadata: dict[str, Any], output_dir: Path) -> None:
    """Write a human-readable experiment report."""
    best = metrics.iloc[0]
    metrics_table = metrics.to_markdown(index=False, floatfmt=".4f")
    report = f"""# Checkpoint 3 Experiments

## Fixed Experiment Context

- Dataset: `{metadata["input_path"]}`
- Dataset SHA-256: `{metadata["dataset_sha256"]}`
- Shape: {metadata["dataset_shape"]["rows"]} rows x {metadata["dataset_shape"]["columns"]} columns
- Selected features: {metadata["selected_feature_count"]}
- Target: `{metadata["target"]}`
- Split: 60/20/20, stratified by `rooms_count`, seed={RANDOM_STATE}
- Baseline: B2 median `price_per_sqm` by `district + rooms_count`, R2={BASELINE_R2_PER_SQM:.4f}

## Model Comparison

{metrics_table}

## Best Model

- Best by test R2 on price_per_sqm: `{best["model"]}`
- Test R2: {best["test_r2_per_sqm"]:.4f}
- Test MAE: {best["test_mae_per_sqm"]:,.0f} RUB/m2
- Test MAPE: {best["test_mape_percent"]:.1f}%

## Feature Set

```text
{chr(10).join(metadata["selected_features"])}
```

## Reproducibility Artifacts

- Metrics CSV: `data/experiments/checkpoint3_metrics.csv`
- Metadata JSON: `data/experiments/checkpoint3_metadata.json`
- Trained models: `data/models/ridge_linear.pkl`, `data/models/random_forest.pkl`, `data/models/gradient_boosting.pkl`
"""
    (output_dir / "checkpoint3_experiment_report.md").write_text(report, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Checkpoint 3 modeling experiments.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    args = parser.parse_args()

    metrics, metadata = run_experiments(args.input, args.output_dir, args.model_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(args.output_dir / "checkpoint3_metrics.csv", index=False)
    (args.output_dir / "checkpoint3_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_report(metrics, metadata, args.output_dir)

    print("\n=== Checkpoint 3 Results ===")
    print(metrics.to_string(index=False))
    print(f"\nSaved metrics to {args.output_dir / 'checkpoint3_metrics.csv'}")
    print(f"Saved metadata to {args.output_dir / 'checkpoint3_metadata.json'}")


if __name__ == "__main__":
    main()
