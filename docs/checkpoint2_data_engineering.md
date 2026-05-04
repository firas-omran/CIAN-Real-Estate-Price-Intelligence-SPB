# Checkpoint 2 - Data Engineering And Data Pipeline

## 1. Data Pipeline

### Approach: ETL

We use **ETL** rather than ELT.

Reason:
- CIAN is an external and unstable web source;
- raw parser output can contain duplicate listings, encoded missing values, outliers, and schema changes;
- model-ready storage should receive only validated and normalized data;
- the project is small enough that local file-based ETL is simpler and reproducible.

### Pipeline Stages

| Stage | Input | Output | Code |
|---|---|---|---|
| Extract | CIAN listing pages | raw CSV snapshots | `src/data/collect_cian_spb.py` |
| Normalize | raw parser output | stable normalized schema | `src/data/collect_cian_spb.py` |
| Validate | normalized/clean data | contract report | `src/data/contract_cian.py` |
| Transform / Clean | normalized CSV | clean dataset | `src/data/clean_cian.py` |
| Feature Engineering | clean dataset | offline features + aggregate tables | `src/features/build_features.py` |
| Sampling | clean dataset | balanced sample by room segment | `src/data/sampling.py` |
| EDA / Reports | clean dataset | figures and summaries | `src/data/make_cian_eda.py` |

The full pipeline can be run with:

```bash
python -m src.pipeline.run_data_pipeline
```

To collect a fresh snapshot first:

```bash
python -m src.pipeline.run_data_pipeline --collect --pages 10 --timeout 20
```

Current pipeline run:

| Artifact | Shape |
|---|---:|
| Normalized snapshot | 1359 rows x 24 columns |
| Clean dataset | 1304 rows x 27 columns |
| Offline feature table | 1304 rows x 39 columns |
| District aggregate table | 22 rows x 6 columns |
| District + rooms aggregate table | 93 rows x 7 columns |
| Underground aggregate table | 73 rows x 6 columns |
| Room segment aggregate table | 5 rows x 6 columns |
| Rooms aggregate table | 5 rows x 6 columns |
| Balanced sample | 1275 rows x 27 columns |

### Tooling Rationale

| Tool | Why |
|---|---|
| `cianparser` | existing parser for CIAN listings, sufficient for educational MVP |
| `pandas` | simple local ETL, aggregation, and reproducible CSV outputs |
| CSV snapshots | transparent, easy to inspect, no infrastructure cost |
| Markdown docs | directly usable in checkpoint presentation |
| Python modules | reproducible CLI pipeline instead of manual notebook-only work |

## 2. Feature Engineering

### Listing-Level Features

| Feature | Source | Refresh | Offline | Online |
|---|---|---|---:|---:|
| `room_segment` | collector segment | weekly / request | yes | yes |
| `rooms_count` | CIAN listing | weekly / request | yes | yes |
| `total_meters` | CIAN listing | weekly / request | yes | yes |
| `log_total_meters` | derived | weekly / request | yes | yes |
| `floor`, `floors_count` | CIAN listing | weekly / request | yes | yes |
| `floor_ratio` | `floor / floors_count` | weekly / request | yes | yes |
| `is_first_floor`, `is_last_floor` | derived | weekly / request | yes | yes |
| `district` | CIAN listing | weekly / request | yes | yes |
| `underground` | CIAN listing | weekly / request | yes | yes |
| `residential_complex` | CIAN listing | weekly / request | yes | cautiously |

### Market Aggregate Features

Aggregation window: current fresh CIAN snapshot.  
Expected refresh: weekly for MVP, daily in a production version.

| Feature | Aggregation | Use |
|---|---|---|
| `district_ads_count` | count by district | market liquidity / supply signal |
| `district_median_price_per_sqm` | median price per m2 by district | local market level |
| `district_rooms_ads_count` | count by district and rooms | comparable supply |
| `district_rooms_median_price_per_sqm` | median price per m2 by district and rooms | comparable price anchor |
| `underground_median_price_per_sqm` | median price per m2 by metro station | transport/location proxy |
| `room_segment_median_price_per_sqm` | median price per m2 by collector room segment | studio-aware room fallback |
| `rooms_median_price_per_sqm` | median price per m2 by rooms | fallback aggregate |

Important leakage rule:

Target-derived aggregates must be computed on the training split only during
model experiments. For online inference they are computed from the latest
historical market snapshot.

## 3. Feature Store Concept

We use a lightweight file-based Feature Store concept for the MVP.

### Offline Feature Store

Path:

```text
data/features/cian_spb_offline_features.csv
```

Purpose:
- model training;
- baseline experiments;
- reproducible feature matrix;
- includes target columns `price` and `log_price`.

### Online Feature Store / Lookup Tables

Paths:

```text
data/features/cian_spb_district_market_aggregates.csv
data/features/cian_spb_district_rooms_market_aggregates.csv
data/features/cian_spb_underground_market_aggregates.csv
data/features/cian_spb_rooms_count_market_aggregates.csv
```

Purpose:
- fast lookup by `district`, `rooms_count`, and `underground`;
- serving-time market context features;
- future API integration.

### Registry

Feature registry:

```text
docs/feature_registry.md
data/features/feature_registry.json
```

The registry records source, type, refresh frequency, offline/online
availability, and leakage risk.

## 4. Data Work: Sampling

Requirement satisfied: **sampling / imbalance correction**.

Problem:
- 1-room apartments dominate the collected snapshot.

Solution:
- create a stratified balanced sample by `rooms_count`;
- each room segment is downsampled to the size of the smallest segment.

Artifacts:

```text
data/processed/cian_spb_balanced_sample.csv
data/processed/sampling_report.md
```

Current balanced room distribution:

| Rooms | Rows |
|---:|---:|
| 0 | 255 |
| 1 | 255 |
| 2 | 255 |
| 3 | 255 |
| 4 | 255 |

This dataset is useful for sanity checks and fair comparison across room
segments, while the full clean dataset remains the main training candidate.

## 5. Checkpoint 2 Artifacts

| Artifact | Path |
|---|---|
| Pipeline runner | `src/pipeline/run_data_pipeline.py` |
| Feature builder | `src/features/build_features.py` |
| Sampling script | `src/data/sampling.py` |
| Offline feature table | `data/features/cian_spb_offline_features.csv` |
| Online aggregate tables | `data/features/cian_spb_*_market_aggregates.csv` |
| Feature registry | `docs/feature_registry.md` |
| DFD | `docs/dfd_checkpoint2.md` |
