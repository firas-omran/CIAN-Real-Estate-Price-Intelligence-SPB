# Data Pipeline Report

## Artifacts

| Artifact | Rows | Columns |
|---|---:|---:|
| `data/raw/cian_spb_normalized_20260531_102418.csv` | 2539 | 24 |
| `data/processed/cian_spb_clean.csv` | 2202 | 27 |
| `data/processed/cian_spb_clean_geo.csv` | 2202 | 36 |
| `data/features/cian_spb_offline_features.csv` | 2202 | 50 |
| `data/features/cian_spb_district_market_aggregates.csv` | 18 | 6 |
| `data/features/cian_spb_district_rooms_market_aggregates.csv` | 90 | 7 |
| `data/features/cian_spb_underground_market_aggregates.csv` | 74 | 6 |
| `data/features/cian_spb_room_segment_market_aggregates.csv` | 5 | 6 |
| `data/features/cian_spb_rooms_market_aggregates.csv` | 5 | 6 |
| `data/processed/cian_spb_balanced_sample.csv` | 1955 | 27 |

## Clean Data Summary

- Data Contract status: OK if `python -m src.data.contract_cian data/processed/cian_spb_clean.csv` passes.
- Clean rows: 2202
- Median price: 16,509,019 RUB
- Median area: 57.2 m2

Room distribution:

rooms_count
0    423
1    480
2    454
3    391
4    454

Room segment distribution:

room_segment
1room     480
2rooms    454
3rooms    391
4rooms    454
studio    423

## Metro Routing

- Metro route distance coverage: 2140 / 2202
- Median route distance to metro: 2.01 km
- Median route duration to metro: 24.1 min

## Sampling

Balanced sample distribution:

rooms_count
0    391
1    391
2    391
3    391
4    391

## Feature Store

- Offline features: `data/features/cian_spb_offline_features.csv`
- Online lookup tables: district, district+rooms, underground, room_segment, rooms.
- Registry: `docs/feature_registry.md` and `data/features/feature_registry.json`.
