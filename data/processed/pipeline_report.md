# Data Pipeline Report

## Artifacts

| Artifact | Rows | Columns |
|---|---:|---:|
| `data/processed/cian_spb_clean.csv` | 1300 | 27 |
| `data/processed/cian_spb_clean_geo.csv` | 1300 | 36 |
| `data/features/cian_spb_offline_features.csv` | 1300 | 50 |
| `data/features/cian_spb_district_market_aggregates.csv` | 18 | 6 |
| `data/features/cian_spb_district_rooms_market_aggregates.csv` | 89 | 7 |
| `data/features/cian_spb_underground_market_aggregates.csv` | 73 | 6 |
| `data/features/cian_spb_room_segment_market_aggregates.csv` | 5 | 6 |
| `data/features/cian_spb_rooms_market_aggregates.csv` | 5 | 6 |
| `data/processed/cian_spb_balanced_sample.csv` | 1275 | 27 |

## Clean Data Summary

- Data Contract status: OK if `python -m src.data.contract_cian data/processed/cian_spb_clean.csv` passes.
- Clean rows: 1300
- Median price: 17,700,000 RUB
- Median area: 58.0 m2

Room distribution:

rooms_count
0    263
1    258
2    261
3    255
4    263

Room segment distribution:

room_segment
1room     258
2rooms    261
3rooms    255
4rooms    263
studio    263

## Metro Routing

- Metro route distance coverage: 1253 / 1300
- Median route distance to metro: 1.76 km
- Median route duration to metro: 21.1 min

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
