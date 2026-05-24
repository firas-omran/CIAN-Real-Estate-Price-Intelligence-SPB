# Checkpoint 3 Experiments

## Fixed Experiment Context

- Dataset: `data/features/cian_spb_offline_features.csv`
- Dataset SHA-256: `0c6708811b5a46cd786a887e44f8bd7368b1841d42276fbf3d9836aad15cade6`
- Shape: 1300 rows x 50 columns
- Selected features: 20
- Target: `log_target_price_per_sqm`
- Split: 60/20/20, stratified by `rooms_count`, seed=42
- Baseline: B2 median `price_per_sqm` by `district + rooms_count`, R2=0.3137

## Model Comparison

| model              | model_class               |   cv_r2_log_mean |   cv_r2_log_std |   val_r2_per_sqm |   val_mae_per_sqm |   val_mape_percent |   test_r2_per_sqm |   test_mae_per_sqm |   test_mape_percent |   improvement_vs_baseline_r2 |
|:-------------------|:--------------------------|-----------------:|----------------:|-----------------:|------------------:|-------------------:|------------------:|-------------------:|--------------------:|-----------------------------:|
| gradient_boosting  | GradientBoostingRegressor |           0.7632 |          0.0188 |           0.6693 |        83559.2379 |            18.9547 |            0.7325 |         80945.2603 |             18.0866 |                       0.4187 |
| random_forest      | RandomForestRegressor     |           0.7441 |          0.0183 |           0.6749 |        79774.0956 |            18.5546 |            0.6901 |         85095.8826 |             18.8828 |                       0.3764 |
| ridge_linear       | Ridge                     |           0.7069 |          0.0252 |           0.4885 |       104331.3751 |            22.8420 |            0.6326 |         96085.7997 |             21.5298 |                       0.3189 |
| B2_non_ml_baseline | group_median_baseline     |         nan      |        nan      |         nan      |          nan      |           nan      |            0.3137 |           nan      |            nan      |                       0.0000 |

## Best Model

- Best by test R2 on price_per_sqm: `gradient_boosting`
- Test R2: 0.7325
- Test MAE: 80,945 RUB/m2
- Test MAPE: 18.1%

## Feature Set

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

## Reproducibility Artifacts

- Metrics CSV: `data/experiments/checkpoint3_metrics.csv`
- Metadata JSON: `data/experiments/checkpoint3_metadata.json`
- Trained models: `data/models/ridge_linear.pkl`, `data/models/random_forest.pkl`, `data/models/gradient_boosting.pkl`
