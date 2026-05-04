# Data Pipeline Report

## Artifacts

| Artifact | Rows | Columns |
|---|---:|---:|
| `data/raw/cian_spb_normalized_20260504_224811.csv` | 1359 | 24 |
| `data/processed/cian_spb_clean.csv` | 1304 | 27 |
| `data/features/cian_spb_offline_features.csv` | 1304 | 39 |
| `data/features/cian_spb_district_market_aggregates.csv` | 22 | 6 |
| `data/features/cian_spb_district_rooms_market_aggregates.csv` | 93 | 7 |
| `data/features/cian_spb_underground_market_aggregates.csv` | 73 | 6 |
| `data/features/cian_spb_room_segment_market_aggregates.csv` | 5 | 6 |
| `data/features/cian_spb_rooms_market_aggregates.csv` | 5 | 6 |
| `data/processed/cian_spb_balanced_sample.csv` | 1275 | 27 |

## Clean Data Summary

- Data Contract status: OK if `python -m src.data.contract_cian data/processed/cian_spb_clean.csv` passes.
- Clean rows: 1304
- Median price: 17,632,490 RUB
- Median area: 58.0 m2

Room distribution:

rooms_count
0    264
1    260
2    262
3    255
4    263

Room segment distribution:

room_segment
1room     260
2rooms    262
3rooms    255
4rooms    263
studio    264

## Sampling

Balanced sample distribution:

rooms_count
0    255
1    255
2    255
3    255
4    255

## Feature Store

- Offline features: `data/features/cian_spb_offline_features.csv`
- Online lookup tables: district, district+rooms, underground, room_segment, rooms.
- Registry: `docs/feature_registry.md` and `data/features/feature_registry.json`.
