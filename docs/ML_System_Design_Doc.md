# ML System Design Doc - CIAN Real Estate Price Intelligence

## 1. Goals And Context

**Project topic:** ML system for estimating the market price of apartments in Saint Petersburg using fresh CIAN listings.

**Business goal:** help buyers, sellers, and analysts estimate whether a listing price is realistic for a given apartment segment and local market context.

**ML task:** supervised regression. The active target is `target_price_per_sqm = price / total_meters` and the modeling variable is `log_target_price_per_sqm = log1p(target_price_per_sqm)`. Per-sqm normalization removes the dominant `total_meters` regressor from the target so model improvements reflect understanding of the local market rather than mechanical scaling by area. The final price is reconstructed at inference as `expm1(prediction) × total_meters` for business metrics.

**Object:** apartment listing from CIAN.

**Pilot context:** public web app where a user enters apartment parameters and receives estimated price, price interval, and comparable market statistics.

**Expected effect:**
- faster rough market valuation;
- transparent comparison with similar listings;
- early detection of unusual or overpriced listings.

## 2. Constraints

| Constraint | Decision |
|---|---|
| Data | Fresh CIAN snapshot collected by parser; no Kaggle data |
| Geography | Saint Petersburg MVP |
| Cost | Free local development and free/low-cost hosting |
| Response time | API target under 500 ms for a single prediction |
| Scale | Demo workload: under 10 RPS |
| Legal/ethical | Limited educational collection, no phone numbers or personal contact data |
| Freshness | Weekly data refresh for MVP, freshness alert if older than 7 days |

## 3. Data Source

**Source:** CIAN apartment sale listings for Saint Petersburg.

**Collection method:** `cianparser` with controlled page count and request timeout.

**Current snapshot:** fresh listings collected on 2026-05-05 local time from CIAN pages available at collection time. This is not a publication-date guarantee; it is a collection timestamp.

**Current cleaned sample:** 1300 listings after validation, cleaning, and the broken-row filter (`district` must be in the official 18-district whitelist `VALID_SPB_DISTRICTS`). Source: 1400 raw rows → 1359 normalized → 1304 cleaned → 1300 after broken-row filter. Studio listings are preserved as `room_segment=studio` and `rooms_count=0`.

**Raw storage:** `data/raw/cian_spb_raw_*.csv`.

**Normalized storage:** `data/raw/cian_spb_normalized_*.csv`.

**Clean storage:** `data/processed/cian_spb_clean.csv`.

## 4. Data Contract

| Field | Required | Range / Values | Max Missing | Notes |
|---|---:|---|---:|---|
| `listing_id` | yes | unique CIAN id | 1% | technical id, not model feature |
| `url` | yes | valid URL | 0% | lineage, not model feature |
| `source` | yes | `cian` | 0% | source tracking |
| `collected_at` | yes | timestamp | 0% | freshness control |
| `location_query` | yes | `Санкт-Петербург` | 0% | MVP geography |
| `price` | yes | 1M-600M RUB | 0% | reconstruction target only; not a model feature |
| `target_price_per_sqm` | yes | 50k-3M RUB/m2 | 0% | active training target (`price / total_meters`) |
| `log_target_price_per_sqm` | yes | finite log1p of target | 0% | training-loss target |
| `total_meters` | yes | 10-500 m2 | 0% | core feature, also used to reconstruct price |
| `rooms_count` | yes | 0-10 | 2% | studio encoded as 0 if parser returns it |
| `floor` | no | 1-100 | 25% | optional |
| `floors_count` | no | 1-100 | 25% | optional |
| `district` | yes | one of 18 SPB official districts | 0% (broken rows filtered) | local market segment, whitelist enforced |
| `underground` | yes | metro station name or `unknown` | 0% literal; ~3.6% are `unknown` | transport proxy |
| `lat` | yes | 59.5-60.3 | 0% (geocoder always falls back to district centroid) | geographic coordinate |
| `lon` | yes | 29.4-30.9 | 0% | geographic coordinate |
| `geo_precision` | yes | one of `house`/`street`/`district` | 0% | precision tier of the geocoded coordinate |
| `distance_to_center_km` | yes | 0-50 km | 0% | haversine to Дворцовая (59.9386, 30.3141) |
| `distance_to_metro_km` | no | 0-15 km | ~3.6% (when underground is `unknown`) | haversine to the named metro station |
| `distance_to_metro_route_km` | no | 0-70 km | up to 100% if routing API is not used | walking route distance from openrouteservice |
| `duration_to_metro_route_min` | no | 0-720 min | up to 100% if routing API is not used | walking route duration from openrouteservice |
| `metro_known` | yes | bool | 0% | flag indicating whether `distance_to_metro_km` is meaningful |
| `residential_complex` | no | string | 70% | useful but leakage/overfit risk |

## 5. Baselines And Metrics

**Baseline solutions without ML:**
- B0 global median `price_per_sqm` × `total_meters`;
- B1 median `price_per_sqm` by room count × `total_meters`;
- B2 median `price_per_sqm` by district and room count × `total_meters`;
- B3 comparable-listings KNN: same district and room count, nearest areas, median `price_per_sqm` × `total_meters`.

All baselines now operate on the active per-sqm target. Final price is reconstructed and reported with the metric set below.

**Training loss for future ML:** MAE on `log_target_price_per_sqm`. Huber as a robust alternative.

**Business metrics, reported on reconstructed `price`:**
- `MAE`: error in RUB;
- `MAPE`: mean percent error;
- `R² on price`: coefficient of determination, comparable with checkpoint 1 numbers.

**Honest metric, reported on `target_price_per_sqm`:**
- `R² on price_per_sqm`: shows whether the model understands the local market beyond mechanical scaling by area. A model with high `R² on price` but low `R² on price_per_sqm` is mostly riding on `total_meters`.

**Current baseline numbers** (reproducible 80/20 split, seed=42, snapshot 1300 rows, geo-enriched):

| Baseline | MAE, RUB | MAPE | R² on price | R² on price_per_sqm |
|---|---:|---:|---:|---:|
| B0 global median price_per_sqm | 19,447,266 | 43.0% | 0.384 | -0.162 |
| B1 by rooms_count | 18,870,219 | 44.0% | 0.448 | -0.122 |
| B2 by district + rooms_count | 14,512,996 | 33.0% | 0.686 | 0.314 |
| B3 KNN comparable on price_per_sqm | 14,618,127 | 34.1% | 0.681 | 0.301 |
| B4 by center_distance_bin + rooms_count | 16,654,207 | 37.4% | 0.577 | 0.101 |
| B5 by metro_distance_bin + rooms_count | 19,117,685 | 44.8% | 0.425 | -0.140 |

B4 (5 center-distance buckets: 0-3, 3-6, 6-10, 10-15, 15+ km) gives positive R^2 on per_sqm — confirming geo signal exists. B5 (5 metro-distance buckets) is weak on its own — metro proximity matters, but without district context it's insufficient.

**Current ML model results** (60/20/20 split, seed=42, 21 selected features in the offline table, no target-derived aggregates/leakage):

| Model | R^2 on per_sqm | MAE (RUB/m^2) | MAPE |
|---|---:|---:|---:|
| B2 baseline (non-ML) | 0.31 | — | — |
| Ridge (linear) | 0.6266 | 96,353 | 21.8% |

Ridge already improves the honest `R^2 on price_per_sqm` by roughly 2x over
the best non-ML baseline B2. CatBoost is supported by `src/models/train.py`
and is the planned main model candidate for Checkpoint 3, together with
experiment tracking and feature-importance analysis.

## 6. Leakage Analysis

Fields not allowed as model features:
- `price`, `log_price` — original raw target;
- `target_price_per_sqm`, `log_target_price_per_sqm` — active target;
- `observed_price_per_sqm`, `price_per_sqm_eda` — computed from target;
- `listing_id`, `url`, `collected_at`, `source` — metadata;
- `street`, `house_number` — used **only** for the geocoder. They feed the
  Nominatim query at training time and the API request at serving time;
  they never enter the model. The model sees only their numeric derivatives
  (`lat`, `lon`, `distance_to_center_km`, `distance_to_metro_km`).

High-risk features:
- `residential_complex`: can improve quality, but may cause memorization on small data; excluded from the modeling feature set.
- KNN comparable features and market-aggregate medians (`*_median_price_per_sqm`): must be calculated using the training split only when running modeling experiments.

## 7. Architecture

See `docs/architecture_cian.md`.

Main layers:
- data collection;
- raw snapshot storage;
- data contract validation;
- cleaning and feature engineering;
- offline feature store;
- baseline/training pipeline;
- model registry;
- prediction API;
- web UI;
- monitoring and retraining triggers.

## 8. Risks v0

| Risk | Category | Cause | Consequence | Mitigation |
|---|---|---|---|---|
| Parser blocked or broken | Data | CIAN anti-bot or layout changes | No fresh data | request timeout, cached snapshots, documented fallback |
| Stale listings | Data | Snapshot not refreshed | Predictions reflect old market | freshness contract and weekly refresh |
| Duplicate/reposted listings | Data | Same listing appears in several result segments | Biased EDA/model | deduplicate by `listing_id`/`url` |
| Luxury outliers | Model | Premium properties dominate errors | Unstable MAE/RMSE | log target, robust metrics, outlier flags |
| Missing geo details | Data | Parser may not provide coordinates | Weak spatial features | use district/metro now, add geocoding later |
| Target leakage | Model | price per m2 is derived from target | Inflated validation quality | ban leakage fields from features |
| Overfitting to complex/street | Model | Small sample and high-cardinality identifiers | Poor generalization | regularization, grouped validation, feature review |
| Free hosting limits | Infrastructure | limited CPU/RAM/cold starts | slow demo or crash | small model, precomputed features, warm-up before defense |
| Distribution drift | Operation | market changes over time | degraded estimates | drift monitoring and retraining trigger |

## 9. Checkpoint Roadmap

**Checkpoint 1:** fresh data source, EDA, contract, architecture, baselines, risks.

**Checkpoint 2:** ETL pipeline, feature registry, offline/online features, DFD. Implemented artifacts:
`src/pipeline/run_data_pipeline.py`, `src/features/build_features.py`, `docs/feature_registry.md`,
`docs/dfd_checkpoint2.md`, and balanced sample by `rooms_count`.

**Checkpoint 3:** train at least two model classes, experiment tracking, retraining strategy.

**Checkpoint 4:** deploy web app/API, monitoring, alerts, runbook, degradation demo.
