# Checkpoint 3. Modeling And Experiments

**Project:** CIAN Real Estate Price Intelligence SPB  
**Goal:** compare multiple ML model classes against the non-ML baseline and define a reproducible retraining strategy.  
**Experiment date:** 24 May 2026  

---

## 1. Models

### Problem Setup

The task is supervised regression for apartment market valuation in Saint Petersburg.

The model target is:

```text
log_target_price_per_sqm = log1p(price / total_meters)
```

Metrics are reported back on the original `price_per_sqm` scale:

```text
predicted_price_per_sqm = expm1(model_prediction)
```

Primary metric:

```text
R2 on price_per_sqm
```

Additional metrics:

```text
MAE, RUB/m2
MAPE, %
```

### Baseline

The main non-ML baseline is B2:

```text
median price_per_sqm by district + rooms_count
```

Baseline score:

```text
R2 on price_per_sqm = 0.3137
```

### Trained Model Classes

Three model classes were trained on the same fixed split.

| Model | Class | Why included |
|---|---|---|
| Ridge | Linear model | simple, stable, interpretable ML baseline |
| RandomForestRegressor | Bagging / tree ensemble | captures non-linear interactions without strong parametric assumptions |
| GradientBoostingRegressor | Boosting / additive trees | strong tabular-data candidate and main current model |

CatBoost is supported by the older `src/models/train.py` script, but the current reproducible Checkpoint 3 experiment uses only scikit-learn models so it can run without extra dependencies.

---

## 2. Experiments

### Reproducibility Contract

Experiment runner:

```text
src/models/experiments.py
```

Artifacts:

```text
data/experiments/checkpoint3_metrics.csv
data/experiments/checkpoint3_metadata.json
data/experiments/checkpoint3_experiment_report.md
data/models/ridge_linear.pkl
data/models/random_forest.pkl
data/models/gradient_boosting.pkl
```

Fixed dataset:

```text
data/features/cian_spb_offline_features.csv
```

Dataset hash:

```text
0c6708811b5a46cd786a887e44f8bd7368b1841d42276fbf3d9836aad15cade6
```

Dataset shape:

```text
1300 rows x 50 columns
```

Selected model features:

```text
20
```

Split:

```text
train / validation / test = 60 / 20 / 20
train rows = 780
validation rows = 260
test rows = 260
stratify = rooms_count
random_state = 42
```

Cross-validation:

```text
5-fold KFold on train split, shuffle=True, random_state=42
```

### Feature Set

Leakage columns are excluded:

```text
price
log_price
target_price_per_sqm
log_target_price_per_sqm
listing_id
url
source
collected_at
location_query
residential_complex
```

Target-derived market aggregates are also excluded from model training.

Final selected features:

```text
rooms_count
total_meters
log_total_meters
floor
floors_count
floor_ratio
is_first_floor
is_last_floor
lat
lon
distance_to_center_km
distance_to_metro_route_km
duration_to_metro_route_min
metro_known
author_type
room_segment
district
underground
geo_precision
metro_route_provider
```

Numerical features use median imputation. Ridge additionally uses standard scaling. Categorical features use constant `unknown` imputation and one-hot encoding.

### Model Parameters

Ridge:

```text
alpha = 1.0
```

RandomForestRegressor:

```text
n_estimators = 400
max_depth = 14
min_samples_leaf = 3
random_state = 42
n_jobs = 1
```

GradientBoostingRegressor:

```text
n_estimators = 350
learning_rate = 0.04
max_depth = 3
min_samples_leaf = 4
subsample = 0.9
random_state = 42
```

### Results

| Model | Class | CV R2 log mean | Validation R2 | Test R2 | Test MAE, RUB/m2 | Test MAPE | R2 gain vs baseline |
|---|---|---:|---:|---:|---:|---:|---:|
| GradientBoostingRegressor | Boosting | 0.7632 | 0.6693 | **0.7325** | **80 945** | **18.1%** | **+0.4187** |
| RandomForestRegressor | Tree ensemble | 0.7441 | 0.6749 | 0.6901 | 85 096 | 18.9% | +0.3764 |
| Ridge | Linear | 0.7069 | 0.4885 | 0.6326 | 96 086 | 21.5% | +0.3189 |
| B2 baseline | Non-ML median | - | - | 0.3137 | - | - | 0.0000 |

Best model:

```text
GradientBoostingRegressor
```

It improves the honest `R2 on price_per_sqm` from `0.3137` to `0.7325`.

### Interpretation

Ridge already provides a strong improvement over the non-ML median baseline, which confirms that the engineered features carry useful signal. Tree ensembles improve further because apartment pricing is non-linear: district, area, floor, metro distance, route distance, and author type interact in ways that a purely linear model cannot fully capture.

The best current candidate for Checkpoint 4 serving is `gradient_boosting.pkl`, because it has the strongest test R2 and the lowest MAE/MAPE among the reproducible experiments.

---

## 3. Retraining Strategy

### Automation Status

Retraining is now partially automated by:

```text
src/pipeline/auto_retrain.py
```

The script evaluates quality and drift triggers, writes a decision report, and
runs `src.models.experiments` automatically when at least one trigger fires.
It does not silently promote a model to production; it records the decision and
retraining outcome so the model can be reviewed before deployment.

Automation artifacts:

```text
data/experiments/auto_retrain_decision.json
data/experiments/auto_retrain_history.jsonl
```

### Retraining Triggers

| Trigger | How to measure | Threshold |
|---|---|---|
| Data freshness drift | max age of `collected_at` | snapshot older than 7 days |
| Target drift | median `target_price_per_sqm` vs training snapshot | change > 15% |
| Area distribution drift | median / p90 `total_meters` | change > 15% |
| Segment drift | max share change by `rooms_count` | share change > 10 pp |
| District distribution drift | max share change by `district` | share change > 10 pp |
| Geo quality degradation | geocoded rows coverage | below 95% |
| Metro route degradation | route distance coverage | below 80% of metro-known listings |
| Metric degradation | best saved experiment R2 | below 0.60 |
| Error degradation | best saved experiment MAPE | above 25% |
| Schema drift | Data Contract failure | any required-field/range error |

Current automated decision example:

```text
decision = retrain
fired_trigger = snapshot_age_days
snapshot_age_days = 19.63
threshold = 7
retrain_status = completed
```

### Action Plan When Trigger Fires

1. Freeze the current production model and record its metrics.
2. Collect a fresh CIAN snapshot.
3. Run the full data pipeline:

```bash
python -m src.pipeline.run_data_pipeline --with-routing
```

4. Validate the data contract:

```bash
python -m src.data.contract_cian data/processed/cian_spb_clean_geo.csv
```

5. Rebuild offline features and route cache.
6. Run all Checkpoint 3 experiments:

```bash
python -m src.models.experiments
```

7. Compare candidate models against:

- previous production model;
- B2 non-ML baseline;
- previous dataset hash and feature set.

8. Accept a new model only if:

- Data Contract passes;
- no leakage columns enter training;
- route/geocoding coverage is acceptable;
- candidate model improves or does not materially degrade R2/MAE/MAPE;
- metrics are stable across validation and test splits.

9. Save the accepted model and metadata:

```text
data/models/
data/experiments/checkpoint3_metadata.json
data/experiments/checkpoint3_metrics.csv
```

10. If no candidate passes, keep the previous model and investigate:

- parser quality;
- drift by district/rooms;
- route API failures;
- outliers/luxury segment;
- data leakage or schema issues.

### Retraining Schedule

For the MVP:

```text
weekly retraining after fresh CIAN snapshot
```

For production:

```text
daily data refresh + weekly retraining, with emergency retraining on drift/metric triggers
```

---

## Commands

Run route-enriched data pipeline:

```bash
python -m src.pipeline.run_data_pipeline --with-routing
```

Run modeling experiments:

```bash
python -m src.models.experiments
```

Run automated retraining decision:

```bash
python -m src.pipeline.auto_retrain
```

Dry-run without executing retraining:

```bash
python -m src.pipeline.auto_retrain --dry-run
```

Inspect metrics:

```bash
cat data/experiments/checkpoint3_metrics.csv
```

Validate data:

```bash
python -m src.data.contract_cian data/processed/cian_spb_clean_geo.csv
```

Expected current best result:

```text
best_model = gradient_boosting
test_r2_per_sqm = 0.7307
test_mae_per_sqm = 80 728 RUB/m2
test_mape = 18.1%
```
