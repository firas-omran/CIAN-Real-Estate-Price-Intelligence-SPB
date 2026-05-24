"""ML model training for CIAN price-per-sqm prediction (Checkpoint 3).

Target: log_target_price_per_sqm
Models: Ridge (linear baseline), CatBoost (main)
Train/val/test: 60/20/20 stratified by rooms_count
Primary metric: R^2 on price_per_sqm (honest metric, target is B2 at 0.31)
"""

from __future__ import annotations

import argparse
import json
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import KFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore", category=FutureWarning)

try:
    from catboost import CatBoostRegressor
except ModuleNotFoundError:  # pragma: no cover - depends on local environment
    CatBoostRegressor = None

DEFAULT_INPUT = Path("data/features/cian_spb_offline_features.csv")
DEFAULT_MODEL_DIR = Path("data/models")
DEFAULT_METRICS_PATH = Path("data/processed/ml_metrics.csv")
RANDOM_STATE = 42
N_SPLITS = 5

# Columns that must never be used as features
LEAKAGE_COLUMNS = {
    "listing_id", "url", "source", "collected_at", "location_query",
    "price", "log_price", "target_price_per_sqm", "log_target_price_per_sqm",
    "residential_complex",  # 70% missing, too sparse for training
}
MODEL_EXCLUDED_COLUMNS = {
    # Use walking-route metro distance instead of the old straight-line estimate.
    "distance_to_metro_km",
}
AGGREGATE_PATTERNS = ("_ads_count", "_median_price", "_median_price_per_sqm",
                      "_p25_price_per_sqm", "_p75_price_per_sqm")

CAT_FEATURES = [
    "room_segment",
    "district",
    "underground",
    "geo_precision",
    "author_type",
    "metro_route_provider",
]

TARGET = "log_target_price_per_sqm"


def _is_aggregate(col: str) -> bool:
    return any(p in col for p in AGGREGATE_PATTERNS)


def prepare_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Select features and target, excluding leakage columns and aggregates."""
    feature_cols = [c for c in df.columns
                    if c not in LEAKAGE_COLUMNS
                    and c not in MODEL_EXCLUDED_COLUMNS
                    and not _is_aggregate(c)]
    X = df[feature_cols].copy()
    y = df[TARGET].copy()
    return X, y


def split_data(X: pd.DataFrame, y: pd.Series) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame,
                                                         pd.Series, pd.Series, pd.Series]:
    """60/20/20 split stratified by rooms_count."""
    stratify_col = X["rooms_count"].astype(int)
    x_train, x_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.4, random_state=RANDOM_STATE, stratify=stratify_col,
    )
    stratify_temp = x_temp["rooms_count"].astype(int)
    x_val, x_test, y_val, y_test = train_test_split(
        x_temp, y_temp, test_size=0.5, random_state=RANDOM_STATE, stratify=stratify_temp,
    )
    return x_train, x_val, x_test, y_train, y_val, y_test


def _numeric_cols(X: pd.DataFrame) -> list[str]:
    return [c for c in X.columns if X[c].dtype in ("float64", "int64", "bool")]


def _cat_cols(X: pd.DataFrame) -> list[str]:
    return [c for c in X.columns if X[c].dtype == "object" or c in CAT_FEATURES]


def build_ridge() -> Pipeline:
    """Ridge regression with scaling and median imputation."""
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    num_features = ["rooms_count", "total_meters", "log_total_meters", "floor",
                    "floors_count", "floor_ratio", "is_first_floor", "is_last_floor",
                    "lat", "lon", "distance_to_center_km",
                    "distance_to_metro_route_km", "duration_to_metro_route_min",
                    "metro_known"]
    cat_features = ["room_segment", "district", "underground", "geo_precision", "author_type", "metro_route_provider"]

    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="unknown")),
        ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False, max_categories=50)),
    ])

    preprocessor = ColumnTransformer([
        ("num", num_pipe, num_features),
        ("cat", cat_pipe, cat_features),
    ], remainder="drop")

    return Pipeline([
        ("prep", preprocessor),
        ("ridge", Ridge(alpha=1.0, random_state=RANDOM_STATE)),
    ])


def _map_metric_prefix(y_true, y_pred, prefix=""):
    """R^2, MAE, MAPE on per_sqm scale."""
    actual = np.expm1(y_true)
    predicted = np.expm1(y_pred)
    mask = actual > 0
    actual = actual[mask]
    predicted = predicted[mask]
    ape = np.abs(predicted - actual) / actual
    return {
        f"{prefix}r2": r2_score(actual, predicted),
        f"{prefix}mae": mean_absolute_error(actual, predicted),
        f"{prefix}mape": float(ape.mean() * 100),
    }


@dataclass
class CatBoostConfig:
    iterations: int = 1000
    learning_rate: float = 0.05
    depth: int = 6
    l2_leaf_reg: float = 3.0
    early_stopping_rounds: int = 50
    random_seed: int = RANDOM_STATE
    verbose: int = 0


def train_catboost(x_train, y_train, x_val, y_val,
                   cat_features: list[str],
                   config: CatBoostConfig | None = None):
    """Train CatBoostRegressor with validation monitoring."""
    if CatBoostRegressor is None:
        raise ModuleNotFoundError(
            "CatBoost is not installed. Install it with `pip install catboost` "
            "or run this script with `--skip-catboost`."
        )

    if config is None:
        config = CatBoostConfig()

    cat_indices = [x_train.columns.get_loc(c) for c in cat_features if c in x_train.columns]

    model = CatBoostRegressor(
        iterations=config.iterations,
        learning_rate=config.learning_rate,
        depth=config.depth,
        l2_leaf_reg=config.l2_leaf_reg,
        early_stopping_rounds=config.early_stopping_rounds,
        random_seed=config.random_seed,
        verbose=config.verbose,
        loss_function="RMSE",
        eval_metric="RMSE",
    )
    model.fit(
        x_train, y_train,
        cat_features=cat_indices,
        eval_set=(x_val, y_val),
    )
    return model


def cross_val_ridge(X_train: pd.DataFrame, y_train: pd.Series) -> dict:
    """5-fold CV for Ridge."""
    pipe = build_ridge()
    cv = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    scores = cross_validate(
        pipe, X_train, y_train, cv=cv,
        scoring={"r2": "r2", "neg_mae": "neg_mean_absolute_error"},
        n_jobs=1,
    )
    return {
        "cv_r2_log_mean": float(scores["test_r2"].mean()),
        "cv_r2_log_std": float(scores["test_r2"].std()),
    }


def cross_val_catboost(X_train: pd.DataFrame, y_train: pd.Series) -> dict:
    """5-fold CV for CatBoost."""
    if CatBoostRegressor is None:
        raise ModuleNotFoundError(
            "CatBoost is not installed. Install it with `pip install catboost` "
            "or run this script with `--skip-catboost`."
        )
    cat_indices = [X_train.columns.get_loc(c) for c in CAT_FEATURES if c in X_train.columns]
    cv = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

    train_r2, val_r2 = [], []
    for train_idx, val_idx in cv.split(X_train):
        x_tr, x_v = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_tr, y_v = y_train.iloc[train_idx], y_train.iloc[val_idx]

        model = CatBoostRegressor(
            iterations=800, learning_rate=0.05, depth=6,
            random_seed=RANDOM_STATE, verbose=0, loss_function="RMSE",
        )
        model.fit(x_tr, y_tr, cat_features=cat_indices, verbose=0)
        preds = model.predict(x_v)
        train_r2.append(r2_score(y_tr, model.predict(x_tr)))
        val_r2.append(r2_score(y_v, preds))

    return {
        "cv_r2_log_mean": float(np.mean(val_r2)),
        "cv_r2_log_std": float(np.std(val_r2)),
    }


def feature_importance_catboost(model, feature_names: list[str]) -> pd.DataFrame:
    importance = model.get_feature_importance()
    df = pd.DataFrame({"feature": feature_names, "importance": importance})
    return df.sort_values("importance", ascending=False).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train ML models for CIAN price-per-sqm.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--metrics", type=Path, default=DEFAULT_METRICS_PATH)
    parser.add_argument("--skip-catboost", action="store_true")
    args = parser.parse_args()

    catboost_available = CatBoostRegressor is not None
    if not catboost_available and not args.skip_catboost:
        print(
            "CatBoost is not installed; continuing with Ridge only. "
            "Install `catboost` or pass `--skip-catboost` to make this explicit."
        )
        args.skip_catboost = True

    # ── Load & prepare ────────────────────────────────────────────────
    df = pd.read_csv(args.input)
    X, y = prepare_features(df)
    x_train, x_val, x_test, y_train, y_val, y_test = split_data(X, y)

    print(f"Full dataset: {X.shape}")
    print(f"Train: {x_train.shape[0]}  Val: {x_val.shape[0]}  Test: {x_test.shape[0]}")
    print(f"Features: {list(X.columns)}")
    print(f"Target: {TARGET}")
    print()

    # ── Ridge Baseline ────────────────────────────────────────────────
    print("=== Ridge (Linear Baseline) ===")
    ridge_cv = cross_val_ridge(x_train, y_train)
    print(f"CV R^2 (log scale): {ridge_cv['cv_r2_log_mean']:.4f} +/- {ridge_cv['cv_r2_log_std']:.4f}")

    ridge_pipe = build_ridge()
    ridge_pipe.fit(x_train, y_train)
    ridge_test_pred = ridge_pipe.predict(x_test)
    ridge_metrics = _map_metric_prefix(y_test, ridge_test_pred, "ridge_")
    print(f"Test R^2 (per_sqm): {ridge_metrics['ridge_r2']:.4f}")
    print(f"Test MAE (per_sqm): {ridge_metrics['ridge_mae']:,.0f} RUB/m^2")
    print(f"Test MAPE: {ridge_metrics['ridge_mape']:.1f}%")
    print()

    # ── CatBoost ──────────────────────────────────────────────────────
    if not args.skip_catboost:
        print("=== CatBoost ===")
        cb_cv = cross_val_catboost(x_train, y_train)
        print(f"CV R^2 (log scale): {cb_cv['cv_r2_log_mean']:.4f} +/- {cb_cv['cv_r2_log_std']:.4f}")

        model = train_catboost(x_train, y_train, x_val, y_val, CAT_FEATURES)
        cb_test_pred = model.predict(x_test)
        cb_metrics = _map_metric_prefix(y_test, cb_test_pred, "catboost_")
        best_iter = model.get_best_iteration()
        print(f"Best iteration: {best_iter}")
        print(f"Test R^2 (per_sqm): {cb_metrics['catboost_r2']:.4f}")
        print(f"Test MAE (per_sqm): {cb_metrics['catboost_mae']:,.0f} RUB/m^2")
        print(f"Test MAPE: {cb_metrics['catboost_mape']:.1f}%")
        print()
    else:
        cb_metrics = {}

    # ── Feature Importance ────────────────────────────────────────────
    if not args.skip_catboost:
        print("=== Top-15 Feature Importance ===")
        fi = feature_importance_catboost(model, list(x_train.columns))
        for _, row in fi.head(15).iterrows():
            bar = "#" * int(row["importance"] * 60)
            print(f"  {row['feature']:<40s} {row['importance']:.4f} {bar}")

    # ── Baseline comparison ───────────────────────────────────────────
    baseline_r2 = 0.3137  # B2 on price_per_sqm
    print()
    print("=== Baseline Comparison (R^2 on price_per_sqm) ===")
    print(f"  B2 (district+rooms):     {baseline_r2:.4f}")
    print(f"  Ridge:                   {ridge_metrics['ridge_r2']:.4f}")
    if not args.skip_catboost:
        print(f"  CatBoost:                {cb_metrics['catboost_r2']:.4f}")

    # ── Save model & metrics ──────────────────────────────────────────
    args.model_dir.mkdir(parents=True, exist_ok=True)
    if not args.skip_catboost:
        model.save_model(str(args.model_dir / "catboost_price_per_sqm.cbm"))
        print(f"\nModel saved to {args.model_dir / 'catboost_price_per_sqm.cbm'}")

    import joblib
    joblib.dump(ridge_pipe, args.model_dir / "ridge_price_per_sqm.pkl")

    metrics_rows = [
        {"model": "B2_baseline", "r2_per_sqm": baseline_r2, "mae_per_sqm": None, "mape": None},
        {"model": "Ridge", "r2_per_sqm": ridge_metrics["ridge_r2"],
         "mae_per_sqm": ridge_metrics["ridge_mae"], "mape": ridge_metrics["ridge_mape"]},
    ]
    if not args.skip_catboost:
        metrics_rows.append({
            "model": "CatBoost", "r2_per_sqm": cb_metrics["catboost_r2"],
            "mae_per_sqm": cb_metrics["catboost_mae"], "mape": cb_metrics["catboost_mape"],
        })
    pd.DataFrame(metrics_rows).to_csv(args.metrics, index=False)
    print(f"Metrics saved to {args.metrics}")


if __name__ == "__main__":
    main()
