# ML System Design Doc - CIAN Real Estate Price Intelligence

## 1. Goals And Context

**Project topic:** ML system for estimating the market price of apartments in Saint Petersburg using fresh CIAN listings.

**Business goal:** help buyers, sellers, and analysts estimate whether a listing price is realistic for a given apartment segment and local market context.

**ML task:** supervised regression. Target variable is `price` in RUB. Primary modeling target for future ML models is `log1p(price)` because real estate prices have a long right tail.

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

**Current cleaned sample:** 1265 listings after validation and cleaning from 1359 normalized rows and 1400 raw rows.

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
| `price` | yes | 1M-600M RUB | 0% | target |
| `total_meters` | yes | 10-500 m2 | 0% | core feature |
| `rooms_count` | yes | 0-10 | 2% | studio encoded as 0 if parser returns it |
| `floor` | no | 1-100 | 25% | optional |
| `floors_count` | no | 1-100 | 25% | optional |
| `district` | no | string | 35% | local market proxy |
| `underground` | no | string | 45% | transport proxy |
| `residential_complex` | no | string | 70% | useful but leakage/overfit risk |

## 5. Baselines And Metrics

**Baseline solutions without ML:**
- B0 global median price;
- B1 median price per m2 by room count multiplied by area;
- B2 median price per m2 by district and room count multiplied by area;
- B3 comparable-listings KNN baseline: same district and room count, nearest areas, median price per m2.

Current best non-ML baseline is B3 with MdAPE 19.82% and WAPE 36.02% on a reproducible 80/20 split.

**Training loss for future ML:** RMSE/MAE on `log1p(price)`.

**Business metrics:**
- `MdAPE`: main metric, robust percent error;
- `MAPE`: average percent error;
- `MAE`: error in RUB;
- `RMSE`: sensitivity to expensive misses;
- `WAPE`: portfolio-level percent error.

This resolves the earlier ambiguity between optimization loss and business metrics.

## 6. Leakage Analysis

Fields not allowed as model features:
- `price`;
- `observed_price_per_sqm` and `price_per_sqm_eda`, because they are computed from the target;
- `listing_id`, `url`, `collected_at`;
- direct identifiers that allow memorization.

High-risk features:
- `residential_complex`, `street`, `house_number`: can improve quality, but may cause memorization on small data;
- KNN comparable features: must be calculated using train data only, not the full dataset.

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

**Checkpoint 2:** ETL pipeline, feature registry, offline/online features, DFD.

**Checkpoint 3:** train at least two model classes, experiment tracking, retraining strategy.

**Checkpoint 4:** deploy web app/API, monitoring, alerts, runbook, degradation demo.
