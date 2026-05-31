# Checkpoint 3 Experiments

## Fixed Experiment Context

- Dataset: `data/features/cian_spb_offline_features.csv`
- Dataset SHA-256: `cbde6ac6dc2776068c17d0230723cd79c554a44151128aaa1f1fc0af0b7a2917`
- Shape: 2202 rows x 50 columns
- Selected features: 20
- Target: `log_target_price_per_sqm`
- Split: 60/20/20, stratified by `rooms_count`, seed=42
- Baseline: B2 median `price_per_sqm` by `district + rooms_count`, R2=0.3137

## Model Comparison

| model              | model_class               |   cv_r2_log_mean |   cv_r2_log_std |   val_r2_per_sqm |   val_mae_per_sqm |   val_mape_percent |   test_r2_per_sqm |   test_mae_per_sqm |   test_mape_percent |   improvement_vs_baseline_r2 |
|:-------------------|:--------------------------|-----------------:|----------------:|-----------------:|------------------:|-------------------:|------------------:|-------------------:|--------------------:|-----------------------------:|
| random_forest      | RandomForestRegressor     |           0.7315 |          0.0294 |           0.5652 |        81319.2328 |            19.4601 |            0.6820 |         72086.9749 |             16.5784 |                       0.3683 |
| gradient_boosting  | GradientBoostingRegressor |           0.7239 |          0.0229 |           0.5435 |        86340.5654 |            20.7358 |            0.6724 |         76470.2735 |             18.0889 |                       0.3587 |
| ridge_linear       | Ridge                     |           0.6431 |          0.0227 |           0.4744 |        94546.1774 |            22.0143 |            0.5803 |         90381.9347 |             21.5436 |                       0.2666 |
| B2_non_ml_baseline | group_median_baseline     |         nan      |        nan      |         nan      |          nan      |           nan      |            0.3137 |           nan      |            nan      |                       0.0000 |

## Best Model

- Best by test R2 on price_per_sqm: `random_forest`
- Test R2: 0.6820
- Test MAE: 72,087 RUB/m2
- Test MAPE: 16.6%

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
