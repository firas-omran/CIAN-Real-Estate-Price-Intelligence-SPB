# Design Spec — Switch Target To `price_per_sqm` And Add Geo Features

- Date: 2026-05-06
- Author: kurzo
- Status: proposed
- Affects checkpoints: 1, 2 (retroactive update of artifacts and docs)
- Out of scope: checkpoint 3 modeling, checkpoint 4 deployment

## 1. Motivation

The current pipeline predicts raw `price` in RUB. Two problems are pushing us
to revisit the target:

1. `price` is dominated by `total_meters`, which mechanically explains a large
   share of the variance. Any model trained on `price` will look good on `R²`
   even when it does not understand the local market. This is the same
   complaint the neighbour project received — "не ясно, на каком признаке
   строится определение таргета".
2. Spatial information in CIAN listings is currently restricted to
   `district` and `underground` strings. We have no continuous geo signal
   such as distance to the city center or to the listed metro station, even
   though such features are among the strongest price drivers in Saint
   Petersburg. Checkpoint 1 report (`docs/checkpoint1_cian_report.md:146`)
   already commits to "future checkpoints will add geocoding and
   distance-to-center features".

This spec resolves both points without expanding the parser, without
changing the snapshot, and without introducing classification or prediction
intervals. Those are explicitly deferred.

## 2. Goals

- Replace the supervised target with `log1p(price_per_sqm)`.
- Reconstruct `price` from the prediction for business-facing metrics.
- Add four geo features: `lat`, `lon`, `distance_to_center_km`,
  `distance_to_metro_km`.
- Recompute baselines B0..B3 on the new target.
- Update project documentation, the feature registry, and module-level
  `CLAUDE.md` to match.
- Define the user input contract for the future API so the serving layer is
  aligned with the new feature set from day one.

## 3. Non-Goals

- Extending the CIAN parser with `--with-extra-data` fields
  (`year_construction`, `house_material_type`, `kitchen_meters`,
  `living_meters`, `heating_type`). Tracked separately, not in this spec.
- Training new ML models. Baselines only.
- Quantile regression, prediction intervals, conformal prediction.
- Classification "is the price market-rate". Reconsidered as a downstream
  overlay over future regression models in checkpoint 4.
- Increasing the snapshot size. Working on the existing 1304 rows.

## 4. Target Definition

Training target:

```
y_train = log1p(price / total_meters)
```

Inverse transform for inference and metrics:

```
price_per_sqm_pred = expm1(y_pred)
price_pred         = price_per_sqm_pred × total_meters
```

Rationale:

- `log1p` is symmetric in percent terms, which matches how price errors are
  perceived in real estate.
- Per-sqm normalization removes the dominant `total_meters` regressor from
  the target, so model improvements reflect local market understanding
  rather than area scaling.

Forbidden features (strict, enforced in `build_features.py`):

- `price`
- `observed_price_per_sqm`, `price_per_sqm_eda`
- `log_price`, `log_price_per_sqm`
- `listing_id`, `url`, `collected_at`, `source`
- `street`, `house_number` (used only for geocoding lineage)
- raw address strings beyond what feeds the geocoder

## 5. New Geo Features

| Feature | Type | Source |
|---|---|---|
| `lat` | numeric | Nominatim geocoding of `street + house_number + district + "Санкт-Петербург"` |
| `lon` | numeric | same |
| `distance_to_center_km` | numeric | haversine from `(lat, lon)` to `(59.9386, 30.3141)` Дворцовая |
| `distance_to_metro_km` | numeric | haversine from `(lat, lon)` to coordinates of the listed `underground` station |

Center coordinate is fixed at Дворцовая площадь. Metro coordinates are
geocoded once per unique station name in the dataset and persisted in
`data/reference/metro_spb_coords.json`. We do **not** snap to the nearest
metro on the map — we use the metro that the listing itself names. This
matches what a serving-time user typed in the address form and avoids
mismatches between train and serve.

## 6. Geocoder Module

**Path:** `src/features/geocoder.py`

**External library:** `geopy` (added to `requirements.txt`) using the
Nominatim provider. No API key required.

**Public surface:**

```python
def geocode_address(district: str | None,
                    street: str | None,
                    house_number: str | None,
                    city: str = "Санкт-Петербург") -> tuple[float, float, str] | None:
    """Returns (lat, lon, precision) where precision in {"house","street","district"}."""

def geocode_metro(station_name: str) -> tuple[float, float] | None: ...

def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float: ...

def distance_to_center_km(lat: float, lon: float) -> float: ...

def enrich_listing(row: dict) -> dict:
    """Returns lat, lon, geo_precision, distance_to_center_km, distance_to_metro_km."""
```

**Three precision tiers (graceful degradation):**

| Tier | Query sent to Nominatim | When used | Coverage on snapshot |
|---|---|---|---|
| `house` | `house_number, street, district, "Санкт-Петербург"` | full address available | ~85% (1119 rows) |
| `street` | `street, district, "Санкт-Петербург"` | `house_number` is missing | ~14% (181 rows) |
| `district` | district centroid (precomputed in `metro_spb_coords.json` style table) | Nominatim fallback fails | residual |

The chosen tier is recorded as a categorical feature `geo_precision` so
downstream models can learn that house-level coordinates are more
trustworthy than district-level ones.

**Caching:**

- `data/cache/geocode_cache.json` — `{address_key: {lat, lon, precision}}`.
  Address key is a stable lowercase `tier|district|street|house_number|city`
  join, so each precision tier caches independently.
- `data/reference/metro_spb_coords.json` — `{station_name: {lat, lon}}`.
- `data/reference/spb_district_centroids.json` —
  `{district: {lat, lon}}`, computed once from the geocoded snapshot and
  committed.

Cache is read on import, written incrementally, and committed to git so
re-runs are deterministic and free of external calls.

**Rate limit:** 1 request per second, with a polite User-Agent header
identifying the project. Conforms to Nominatim usage policy.

**Failure handling:**

- `district` not in `VALID_SPB_DISTRICTS` → row is **already filtered out**
  by `clean_cian.py` (see §7). Geocoder is never called on broken rows.
- Address resolved at `house` tier → all four geo features populated.
- Address resolved at `street` tier → all four geo features populated, but
  `geo_precision = "street"`.
- Address could not be resolved at any tier → district centroid is used as
  fallback, `geo_precision = "district"`.
- `underground == "unknown"` (47 rows on the current snapshot) →
  `distance_to_metro_km = NaN`, `metro_known = False`. We do NOT impute the
  median, because that would invent a signal that does not exist. Models
  handle the NaN natively (CatBoost) or we apply the district-median impute
  only inside the model's preprocessing pipeline, never in the offline
  feature store.

## 7. Pipeline Changes

The data pipeline gains two explicit changes — a stricter cleaning step
and a new geocoding step between cleaning and feature engineering:

```
collect_cian_spb            → data/raw/cian_spb_raw_*.csv
clean_cian                  → data/processed/cian_spb_clean.csv
   [UPDATED] add a strict filter: district must be in VALID_SPB_DISTRICTS
   (the 18 official Saint Petersburg districts). Removes ~4 broken rows
   on the current snapshot where the parser placed listing titles into
   the district field. Constant VALID_SPB_DISTRICTS is exported from
   src/data/clean_cian.py and reused by EDA, geocoder, and feature
   builder so the canonical district list lives in one place.
[NEW] geocode_clean         → data/processed/cian_spb_clean_geo.csv
   Calls geocoder.enrich_listing on each row with three precision tiers
   (§6). Adds columns lat, lon, geo_precision, distance_to_center_km,
   distance_to_metro_km, metro_known.
make_cian_eda               → data/processed/cian_eda_summary.md
   [UPDATED] Now reports geocoding coverage (broken-row count, unknown-
   metro count, fill rate per geo_precision tier), distribution plots
   for distance_to_center_km and distance_to_metro_km, and a scatter of
   price_per_sqm vs distance_to_center_km that visually demonstrates
   the new spatial signal.
build_features              → data/features/cian_spb_offline_features.csv
   Now includes geo features and the new target log_target_price_per_sqm.
baseline_cian               → data/processed/baseline_metrics.csv
   Recomputed against the new target.
```

`src/pipeline/run_data_pipeline.py` orchestrates the updated order. The
geocoding step is idempotent thanks to the on-disk cache, so EDA and
baseline reruns are fast even after the first 10–15 minute warm-up.

## 8. Baselines Update

`src/models/baseline_cian.py` predicts `price_per_sqm` directly, then
multiplies by `total_meters` for business metrics:

| Baseline | Definition |
|---|---|
| B0 | global median `price_per_sqm` |
| B1 | median `price_per_sqm` by `rooms_count` |
| B2 | median `price_per_sqm` by `district × rooms_count` |
| B3 | KNN-style comparable median on `price_per_sqm` within district + rooms |

Reported metrics, all on a reproducible 80/20 split:

- on reconstructed `price`: `MAE` (RUB), `MAPE` (%), `R²`
- on `price_per_sqm`: `R²` only — used as the *honest* metric that shows
  whether the baseline understands the market beyond area scaling

Existing baseline metrics in
`docs/checkpoint1_cian_report.md` Section 5 are replaced. The historical
table can stay as a footnote for reference.

## 9. User Input Contract (For Future API)

The future `/predict` endpoint accepts five user-typed fields:

```json
{
  "address": "Московский, Пулковское шоссе, 95к4",
  "total_meters": 56.4,
  "rooms_count": 2,
  "floor": 7,
  "floors_count": 12
}
```

Server-side enrichment (same `geocoder.py` module reused at training time):

- parse `address` → call Nominatim → `lat`, `lon`
- reverse-geocode or substring-match → `district`
- nearest-station-by-name (when provided) or kd-tree-by-coords (fallback)
  → `underground`
- compute `distance_to_center_km`, `distance_to_metro_km`
- look up `district_rooms_median_price_per_sqm` and
  `underground_median_price_per_sqm` in the offline feature store

The user never enters latitude or kilometers. Address text is the raw input,
all numeric features are server-derived.

A serving-time fallback for unresolved addresses uses dropdown selectors of
known districts and metro stations from the training snapshot, with the
district centroid as the geo proxy.

## 10. Documentation Updates

| File | Change |
|---|---|
| `docs/ML_System_Design_Doc.md` | §1 task formulation, §3 data contract (add `lat, lon, geo_precision, distance_to_center_km, distance_to_metro_km, metro_known`; tighten `Max Missing` for `district` and `underground` to match observed values), §5 baselines and metrics, §6 leakage analysis (forbid `street, house_number` as model features) |
| `docs/checkpoint1_cian_report.md` | §1 target, §4 leakage, §5 baseline numbers, §8 supervisor feedback addressed (geocoding shipped, target rationale spelled out) |
| `docs/checkpoint2_data_engineering.md` | listing-level features table extended with geo features, pipeline stages list extended with the geocoding step and the EDA refresh |
| `docs/feature_registry.md` and `data/features/feature_registry.json` | six new entries (lat, lon, geo_precision, distance_to_center_km, distance_to_metro_km, metro_known) |
| `src/data/make_cian_eda.py` | new geocoding-coverage section in the report, two new figures (`distance_to_center_distribution.png`, `price_per_sqm_vs_distance.png`), regenerated `cian_eda_summary.md` |
| `README.md` | new pipeline command, target description, mention of geocoding step |

Module-level `CLAUDE.md` files are introduced for `src/data/`,
`src/features/`, `src/models/`, since this iteration touches all three.
Frontmatter `status: in_progress` while the spec is being executed,
`status: done` once the spec is committed.

## 11. Risks Specific To This Change

| Risk | Cause | Mitigation |
|---|---|---|
| Nominatim rate-limit or outage | external dependency | persistent JSON cache, retries with exponential backoff, fallback to district centroid (precision tier `district`) |
| Wrong street disambiguation in СПб | duplicate street names across districts | `district` is part of every geocoding query |
| Mixed geocoding precision skews learning | house-level and district-level coordinates have very different noise | expose `geo_precision` as a categorical feature so the model can downweight low-precision rows |
| Broken parser rows poison spatial features | listing titles ending up in `district`/`street`/`house_number` | strict whitelist filter `VALID_SPB_DISTRICTS` in `clean_cian.py` removes them before geocoding |
| `underground == "unknown"` rows lose metro signal | parser failure on 47 rows | leave `distance_to_metro_km = NaN` and expose `metro_known` flag instead of synthesizing a value |
| Per-sqm target hides non-linearity in area | studios overpriced per sqm, large flats discounted per sqm | keep `total_meters` and `room_segment` as features, evaluate per-segment metrics |
| Metric incomparability with old design doc | switching target without updating tables | this spec mandates updating §5 in design doc and §5 in checkpoint 1 report |

## 12. Acceptance Criteria

Counts below assume the current snapshot (1304 rows pre-clean, 1300 rows
post-clean after the broken-row filter).

- `geocoder.py` exists with documented public surface and at least one
  unit test for `haversine_km`.
- `clean_cian.py` removes broken rows by `VALID_SPB_DISTRICTS`. Output
  `cian_spb_clean.csv` shrinks from 1304 to 1300 rows on this snapshot.
- `data/processed/cian_spb_clean_geo.csv` is generated with **non-null
  `lat`/`lon` for 100% of rows** (using the three precision tiers; tier
  is recorded in `geo_precision`).
- `distance_to_metro_km` is non-null for at least 95% of rows; `metro_known`
  flag is populated for every row.
- `data/features/cian_spb_offline_features.csv` exposes the new target
  column `log_target_price_per_sqm` and the six geo features (`lat`, `lon`,
  `geo_precision`, `distance_to_center_km`, `distance_to_metro_km`,
  `metro_known`).
- `data/processed/baseline_metrics.csv` is regenerated with the new metric
  set: `MAE` (RUB), `MAPE` (%), `R²` on price, `R²` on price_per_sqm.
- `cian_eda_summary.md` reports geocoding coverage and the new figures
  exist in `data/processed/figures/`.
- All documentation files in §10 are updated and consistent with each other.
- Module-level `CLAUDE.md` files exist in `src/data/`, `src/features/`,
  `src/models/`, with frontmatter `status: done` after this iteration is
  committed.
- One commit "Switch target to price_per_sqm and add geo features".

## 13. Out-Of-Scope Reminders

These ideas surfaced during brainstorming and are intentionally deferred:

- Extending the parser via `--with-extra-data` (year_construction, kitchen
  area, house material). Recommended for the next iteration once the geo
  pipeline is stable.
- Quantile regression and prediction intervals for the green/yellow/red
  market-fit indicator. Belongs to checkpoint 4.
- Classification target "is the price market-rate". Synthesizing such a
  label without an external reference would make the task circular against
  baseline B2/B3.
