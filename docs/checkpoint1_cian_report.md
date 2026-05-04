# Checkpoint 1 Report - CIAN Real Estate Price Intelligence

## 1. Problem Statement

**Topic:** ML system for apartment price estimation in Saint Petersburg using fresh CIAN listings.

**Task type:** regression.

**Object:** apartment sale listing.

**Target:** `price`, RUB.

**User value:** estimate market price and compare a listing with similar apartments.

**Core constraints:** fresh data, free project budget, response time under 500 ms for demo API, Saint Petersburg MVP, no personal contact data collection.

## 2. Fresh Data Collection

**Source:** CIAN sale apartment listings.

**Collection script:** `src/data/collect_cian_spb.py`.

**Collection strategy:** separate room segments: studio, 1, 2, 3, 4 rooms. Each segment is collected page-by-page and then merged with duplicate removal.

**Current data files:**
- raw: `data/raw/cian_spb_raw_*.csv`;
- normalized: `data/raw/cian_spb_normalized_*.csv`;
- clean: `data/processed/cian_spb_clean.csv`.

**Freshness:** the current snapshot was collected on 2026-05-05. This means listings were available on CIAN at collection time.

## 3. EDA Summary

Current cleaned dataset:

| Metric | Value |
|---|---:|
| Raw rows | 1400 |
| Raw normalized rows | 1359 |
| Clean rows | 1265 |
| Removed rows | 94 |
| Median price | 18,623,690 RUB |
| Mean price | 37,634,262 RUB |
| Median area | 60.1 m2 |
| Median price per m2 | 340,633 RUB |

Room distribution after cleaning:

| Rooms | Listings |
|---:|---:|
| 1 | 466 |
| 2 | 269 |
| 3 | 260 |
| 4 | 270 |

Generated figures:
- `data/processed/figures/price_distribution.png`;
- `data/processed/figures/log_price_distribution.png`;
- `data/processed/figures/area_distribution.png`;
- `data/processed/figures/price_vs_area.png`;
- `data/processed/figures/rooms_distribution.png`;
- `data/processed/figures/price_by_district_top15.png`;
- `data/processed/figures/price_per_sqm_by_rooms.png`;
- `data/processed/figures/missing_values.png`;
- `data/processed/figures/correlation_matrix.png`.

Expected observations:
- price distribution has a long right tail;
- `log1p(price)` is more stable for modeling;
- area is a strong driver of price;
- district and room count are important market segments;
- floor fields can have missing encoded values and require cleaning.

## 4. Leakage Analysis

Do not use as model features:
- `price`;
- `observed_price_per_sqm`;
- `price_per_sqm_eda`;
- `listing_id`;
- `url`;
- `collected_at`.

Careful features:
- `residential_complex`, `street`, `house_number`: possible memorization;
- KNN comparable features: calculate using train data only.

## 5. Baseline

Implemented in `src/models/baseline_cian.py`.

Baselines:
- B0 global median;
- B1 median price per m2 by rooms multiplied by area;
- B2 median price per m2 by district and rooms multiplied by area;
- B3 comparable-listings KNN baseline.

Metrics:
- MAE;
- RMSE;
- MAPE;
- MdAPE;
- WAPE.

Current baseline results on an 80/20 reproducible split:

| Baseline | MAE, RUB | RMSE, RUB | MAPE | MdAPE | WAPE |
|---|---:|---:|---:|---:|---:|
| B0 global median price | 29,263,930 | 57,890,370 | 72.91% | 63.66% | 72.12% |
| B1 median price/m2 by rooms * area | 20,678,910 | 43,692,770 | 48.78% | 42.54% | 50.96% |
| B2 median price/m2 by district+rooms * area | 15,421,100 | 34,708,800 | 33.63% | 21.22% | 38.01% |
| B3 comparable-listings KNN baseline | 14,614,250 | 33,161,390 | 31.71% | 19.82% | 36.02% |

Conclusion: local market context matters. Moving from a global median to district/rooms and comparable-listing baselines substantially improves error, which supports the supervisor's suggestion to include neighborhood-level and KNN-style information.

## 6. Architecture

The architecture is documented in `docs/architecture_cian.md`.

It contains data source, collector, raw storage, data validation, cleaning, feature engineering, offline feature store, training, model registry, API, UI, monitoring, alerts, and retraining triggers.

## 7. Risks

The risk table is in `docs/ML_System_Design_Doc.md`.

Main risks:
- parser blocking;
- stale data;
- duplicates;
- luxury outliers;
- missing geo information;
- target leakage;
- overfitting to address-like fields;
- hosting limits;
- market drift.

## 8. Supervisor Feedback Addressed

**Architecture:** replaced a linear pipeline with a complete ML system architecture including storage, validation, serving, monitoring, and retraining.

**Loss/metrics:** separated future training loss on `log1p(price)` from business metrics such as MdAPE, MAPE, MAE, RMSE, WAPE.

**Old data:** removed Kaggle dependency and switched to fresh CIAN snapshots.

**Insufficient features:** added district, underground, residential complex, floor ratio, first/last floor flags, and comparable-listings baseline. Future checkpoints will add geocoding and distance-to-center features.
