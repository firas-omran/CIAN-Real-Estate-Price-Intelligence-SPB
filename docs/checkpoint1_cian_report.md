# Checkpoint 1 Report - CIAN Real Estate Price Intelligence

## 1. Problem Statement

**Topic:** ML system for apartment price estimation in Saint Petersburg using fresh CIAN listings.

**Task type:** regression.

**Object:** apartment sale listing.

**Active target:** `target_price_per_sqm = price / total_meters`. Models train on `log_target_price_per_sqm = log1p(target_price_per_sqm)`. Price is reconstructed at inference as `expm1(prediction) × total_meters` for business metrics. This decision was taken in the 2026-05-06 design iteration (`docs/superpowers/specs/2026-05-06-target-switch-and-geo-design.md`) to avoid the situation where a model looks good purely because `total_meters` mechanically explains most of the variance in raw `price`.

**User value:** estimate market price per square meter and the reconstructed total price; compare a listing with similar apartments.

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
| Clean rows (after range filters) | 1304 |
| Clean rows (after broken-row filter, current snapshot) | 1300 |
| Removed rows total | 100 |
| Median price | 17,700,000 RUB |
| Mean price | 36,466,864 RUB |
| Median area | 58.0 m2 |
| Median price per m2 | 339,430 RUB |
| Median distance to center | 6.78 km |
| Median distance to metro (when known) | 1.41 km |
| Geocoded rows | 1300 / 1300 (100%) |
| Geocoding precision: house | 1016 (78.2%) |
| Geocoding precision: street | 118 (9.1%) |
| Geocoding precision: district fallback | 166 (12.8%) |
| Metro distance available | 1253 / 1300 (96.4%) |

Room distribution after cleaning:

| Rooms | Listings |
|---:|---:|
| 0 / studio | 264 |
| 1 | 260 |
| 2 | 262 |
| 3 | 255 |
| 4 | 263 |

Generated figures:
- `data/processed/figures/price_distribution.png`;
- `data/processed/figures/log_price_distribution.png`;
- `data/processed/figures/area_distribution.png`;
- `data/processed/figures/price_vs_area.png`;
- `data/processed/figures/rooms_distribution.png`;
- `data/processed/figures/room_segment_distribution.png`;
- `data/processed/figures/price_by_district_top15.png`;
- `data/processed/figures/price_per_sqm_by_rooms.png`;
- `data/processed/figures/missing_values.png`;
- `data/processed/figures/correlation_matrix.png`;
- `data/processed/figures/distance_to_center_distribution.png`;
- `data/processed/figures/distance_to_metro_distribution.png`;
- `data/processed/figures/price_per_sqm_vs_distance.png`;
- `data/processed/figures/spb_map_price_per_sqm.png`.

Expected observations:
- price distribution has a long right tail;
- `log1p(price)` is more stable for modeling;
- area is a strong driver of price;
- district and room count are important market segments;
- floor fields can have missing encoded values and require cleaning.

## 4. Leakage Analysis

Do not use as model features:
- `price`, `log_price`;
- `target_price_per_sqm`, `log_target_price_per_sqm` (the active target);
- `observed_price_per_sqm`, `price_per_sqm_eda`;
- `listing_id`, `url`, `source`, `collected_at`;
- `street`, `house_number`: used **only** as input to the geocoder (Nominatim) at training and serving time. The model itself sees only their numeric derivatives (`lat`, `lon`, `distance_to_center_km`, `distance_to_metro_km`).

Careful features:
- `residential_complex`: high cardinality on a 1300-row dataset, excluded from the modeling feature set (memorization risk).
- KNN comparable features and market-aggregate medians (`*_median_price_per_sqm`): calculate using the training split only when running modeling experiments.

## 5. Baseline

Implemented in `src/models/baseline_cian.py`.

All baselines now predict `target_price_per_sqm` directly and reconstruct `price = predicted_price_per_sqm × total_meters` for business metrics. Loss inside the baselines is implicit (medians are non-parametric); the metric set is below.

Baselines:
- B0 global median `price_per_sqm`;
- B1 median `price_per_sqm` by rooms;
- B2 median `price_per_sqm` by district and rooms;
- B3 comparable-listings KNN: same district and rooms, nearest by area, median per_sqm;
- B4 median `price_per_sqm` by center distance bucket and rooms;
- B5 median `price_per_sqm` by metro distance bucket and rooms.

Metrics:
- `MAE` (RUB), `MAPE` (%), `R²` on reconstructed `price` — comparable with business reporting;
- `R²` on `price_per_sqm` — the *honest* metric: shows whether the baseline understands the local market beyond mechanical scaling by area.

Current baseline results on a reproducible 80/20 split (seed=42, snapshot 1300 rows, geo-enriched):

| Baseline | MAE, RUB | MAPE | R² on price | R² on price_per_sqm |
|---|---:|---:|---:|---:|
| B0 global median price_per_sqm | 19,447,266 | 43.0% | 0.384 | -0.162 |
| B1 by rooms_count | 18,870,219 | 44.0% | 0.448 | -0.122 |
| B2 by district + rooms_count | 14,512,996 | 33.0% | 0.686 | 0.314 |
| B3 KNN comparable on price_per_sqm | 14,618,127 | 34.1% | 0.681 | 0.301 |
| B4 by center_distance_bin + rooms_count | 16,654,207 | 37.4% | 0.577 | 0.101 |
| B5 by metro_distance_bin + rooms_count | 19,117,685 | 44.8% | 0.425 | -0.140 |

Conclusion. The original target (`price`) made B0 and B1 look stronger than they actually are: their `R² on price` was lifted by the dominance of `total_meters` in the price formula. The honest metric `R² on price_per_sqm` is **negative** for B0 and B1 — these baselines do not understand the market, only its area scaling. Only district-aware baselines (B2, B3) achieve a positive `R² ≈ 0.30` on the per-sqm target. B4 (center distance bins) gives R² = +0.10 — confirming continuous geo signal exists. B5 (metro bins alone) is negative — metro distance needs district context.

ML models (Checkpoint 2/3 boundary): Ridge achieves R² = 0.63 on per_sqm, CatBoost achieves **R² = 0.71** — 2.3× improvement over the best non-ML baseline B2 (0.31). Top features: district (24.7%), author_type (13.7%), distance_to_center_km (7.1%), floors_count (6.5%), total_meters (6.1%).

For comparison, the historical numbers on the original `price` target (ML System Design Doc as committed in checkpoint 1, before the broken-row filter and the target switch) — kept here for context only:

| Baseline (old) | MAE | RMSE | MAPE | MdAPE | WAPE |
|---|---:|---:|---:|---:|---:|
| B0 global median price | 29.2M | 61.0M | 79.3% | 67.0% | 74.5% |
| B1 by rooms × area | 19.4M | 42.7M | 47.9% | 42.4% | 49.5% |
| B2 by district + rooms × area | 14.4M | 34.7M | 31.7% | 20.0% | 36.6% |
| B3 KNN comparable | 13.9M | 30.6M | 32.3% | 20.3% | 35.4% |

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

**Old data:** removed Kaggle dependency and switched to fresh CIAN snapshots collected by `cianparser`.

**Loss/metrics:** training loss decoupled from business reporting. Models train on `log_target_price_per_sqm`. Business metrics are computed on reconstructed `price` (`MAE`, `MAPE`, `R²`). One additional honest metric — `R² on price_per_sqm` — exposes whether the baseline understands the local market or merely scales by `total_meters`. The 2026-05-06 baseline run shows that this honest metric is *negative* for global / rooms-only baselines (B0, B1) and only positive for district-aware baselines (B2, B3) — exactly the diagnostic the original supervisor feedback asked for.

**Insufficient features:** added district, underground, floor ratio, first/last floor flags, and the comparable-listings baseline in checkpoint 1. The 2026-05-06 iteration delivered the geocoding step that was promised here as future work: every listing now carries `lat`, `lon`, `geo_precision`, `distance_to_center_km`, `distance_to_metro_km`, `metro_known`. The geocoder uses Nominatim with a persistent cache and three precision tiers (house / street / district fallback).

**Target rationale:** the active target is `target_price_per_sqm`, not raw `price`. This addresses the "не ясно, на каком признаке строится определение таргета" feedback: per-sqm normalization removes the dominant `total_meters` effect, so each remaining feature has a clear interpretable contribution. `street` and `house_number` participate only in the geocoder (lineage-only), never as model features — the model sees their numeric derivatives instead.

**Broken parser rows:** the cleaning pipeline now enforces `district ∈ VALID_SPB_DISTRICTS` (the 18 official districts). Four broken rows where the parser placed listing titles into the district field are filtered out, so the geocoder is never called on garbage.
