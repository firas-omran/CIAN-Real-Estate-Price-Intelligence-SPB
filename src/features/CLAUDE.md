---
name: src/features
description: Feature engineering, геокодинг адресов, формирование оффлайн feature store и market-aggregate lookup-таблиц для serving.
status: done
owner: kurzo
---

# Модуль `src/features` — feature store и геокодинг

## Назначение

Принимает clean-датасет от `src/data/`, превращает его в матрицу фич,
готовую для baselines и будущих ML-моделей, и в lookup-таблицы для
inference. Здесь же живёт геокодинг — единый модуль, который и при
тренировке, и при serving превращает строковый адрес пользователя в
числовые гео-фичи.

## Контракт

**Вход:**
- `data/processed/cian_spb_clean.csv` (от `src/data/clean_cian.py`)
- адрес от пользователя в API (тот же геокодер, см. `geocoder.py`).

**Выход:**

| Файл | Содержимое |
|---|---|
| `data/processed/cian_spb_clean_geo.csv` | clean-датасет + 6 геофич: `lat, lon, geo_precision, distance_to_center_km, distance_to_metro_km, metro_known` |
| `data/features/cian_spb_offline_features.csv` | offline feature table для тренировки и baselines, включая таргет `log_target_price_per_sqm` |
| `data/features/cian_spb_*_market_aggregates.csv` | lookup-таблицы (district, district_rooms, underground, room_segment, rooms) для serving |
| `data/features/feature_registry.json` + `docs/feature_registry.md` | registry с типами, источниками, leakage-risk |
| `data/cache/geocode_cache.json` | кэш Nominatim, идемпотентность пайплайна |
| `data/reference/metro_spb_coords.json` | координаты ~73 станций метро СПб |
| `data/reference/spb_district_centroids.json` | центроиды 18 районов как fallback для геокодинга |

**Ключевые гарантии:**
- `target_price_per_sqm = price / total_meters`
- `log_target_price_per_sqm = log1p(target_price_per_sqm)` — обучающий таргет
- `distance_to_center_km` от центра Дворцовой (59.9386, 30.3141) — haversine
- `distance_to_metro_km` от станции, **указанной в `underground`** (не «ближайшая»)
- target-derived market aggregates считаются **только на train-сплите** в экспериментах

## Зависимости

**От `src/data/`:** `cian_spb_clean.csv`, константа `VALID_SPB_DISTRICTS` из `src/data/clean_cian.py`.

**Внешние:** `geopy` (Nominatim), `pandas`, `numpy`. Геокодинг — единственная зависимость от интернета; после первого прогона работает из локального кэша.

**Кому передаём:**
- `src/models/baseline_cian.py` — читает `cian_spb_offline_features.csv` и `cian_spb_clean_geo.csv`
- `src/api/main.py` — будет читать lookup-таблицы и вызывать `geocoder.enrich_listing` на запрос пользователя

## Внутреннее устройство

| Файл | Что делает |
|---|---|
| `geocoder.py` | Nominatim-клиент, кэш, haversine, три precision-tier'а (house/street/district), справочник метро, центроиды районов, `enrich_listing(row)` |
| `build_features.py` | склейка clean + геофич + market aggregates, формирование `log_target_price_per_sqm`, запись offline feature table и aggregate lookup'ов, registry |

## Тесты

- `geocoder.py` — юнит-тест на `haversine_km` (известные пары координат), smoke-тест на mock Nominatim для cache hit/miss.
- `build_features.py` — smoke-тест: вызов `build_offline_features(small_df)`, проверка наличия таргет-колонки и геофич, отсутствия leakage-колонок (`price`, `observed_price_per_sqm`).

Команда полного перепрогона:

```bash
python -m src.features.geocoder --input data/processed/cian_spb_clean.csv \
    --output data/processed/cian_spb_clean_geo.csv
python -m src.features.build_features
```

Геокодинг идемпотентен: после первого запуска (~10-15 минут на 1300 строк
с rate-limit 1 req/s) последующие прогоны мгновенные за счёт кэша.

## TODO

- [x] Реализовать `geocoder.py` (2026-05-06): три tier'а, кэш, haversine, seed-станции метро для проблемных кейсов (Девяткино, Маяковская, Достоевская, Зенит).
- [x] Расширить `build_features.py` под новый таргет и геофичи (2026-05-06): добавлены `target_price_per_sqm`, `log_target_price_per_sqm`, шесть геофич; market aggregates считаются на новом таргете; registry обновлён до 17 записей.
- [ ] Добавить prediction intervals (quantile regression) — будущее, чекпоинт 4.
- [ ] Поддержка `with_extra_data` полей (`year_construction`,
  `kitchen_meters`, `living_meters`, `house_material_type`) — отдельная
  итерация, см. §13 спеки.

## Документация и ссылки

- Активная спека: `docs/superpowers/specs/2026-05-06-target-switch-and-geo-design.md`
- ML System Design Doc: `docs/ML_System_Design_Doc.md` §3 (data contract), §6 (leakage)
- Data Engineering report: `docs/checkpoint2_data_engineering.md` §2 (Feature Engineering)
- Feature Registry: `docs/feature_registry.md`
- Geopy / Nominatim docs: <https://geopy.readthedocs.io/>
- Nominatim usage policy: <https://operations.osmfoundation.org/policies/nominatim/>
