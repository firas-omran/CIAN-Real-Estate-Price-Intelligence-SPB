# Отчет по чекпоинтам 1 и 2

**Проект:** CIAN Real Estate Price Intelligence SPB  
**Тема:** ML-система оценки цены за квадратный метр квартир Санкт-Петербурга по свежим объявлениям CIAN  
**Дата актуального прогона:** 24 мая 2026  

---

## Чекпоинт 1. Постановка задачи и первичное проектирование

### 1. Выбор темы и постановка задачи

**Выбранная тема:** прогнозирование рыночной цены за квадратный метр квартиры в Санкт-Петербурге по объявлениям CIAN.

**Тип ML-задачи:** supervised regression.

**Объект:** объявление о продаже квартиры.

**Контекст применения:** пользователь вводит параметры квартиры или анализирует объявление, а система оценивает рыночную цену за м2 и восстановленную полную стоимость квартиры.

**Таргет:**

```text
target_price_per_sqm = price / total_meters
log_target_price_per_sqm = log1p(target_price_per_sqm)
```

Модель обучается на `log_target_price_per_sqm`, а итоговая цена восстанавливается как:

```text
predicted_price = expm1(prediction) * total_meters
```

Такой таргет выбран, чтобы модель не получала искусственно высокий результат только из-за механической связи `price = price_per_sqm * total_meters`.

**Ожидаемый пользовательский/бизнес-эффект:**

- быстрая оценка рыночной стоимости квартиры;
- сравнение объявления с похожими объектами;
- обнаружение переоцененных и недооцененных вариантов;
- подготовка основы для будущего API/веб-интерфейса.

**Ограничения:**

- данные: свежие snapshot'ы CIAN, без Kaggle;
- география: Санкт-Петербург;
- бюджет: бесплатная/локальная разработка;
- latency target для демо API: до 500 мс на один prediction;
- масштаб: demo workload, менее 10 RPS;
- приватность: не собираются телефоны и персональные контакты.

**Baseline без ML:**

Реализованы B0-B5 в `src/models/baseline_cian.py`.

| Baseline | Идея | R2 на price_per_sqm |
|---|---|---:|
| B0 | global median price_per_sqm | -0.162 |
| B1 | median price_per_sqm by rooms | -0.122 |
| B2 | median price_per_sqm by district + rooms | 0.314 |
| B3 | comparable-listings KNN | 0.301 |
| B4 | center distance bucket + rooms | 0.101 |
| B5 | metro distance bucket + rooms | -0.140 |

Лучший non-ML baseline: **B2, R2 = 0.3137** на `price_per_sqm`.

### 2. Первичный сбор данных и EDA

**Источник данных:** CIAN sale apartment listings for Saint Petersburg.

**Способ получения:** `cianparser`, сбор по 5 сегментам комнатности:

```text
studio, 1room, 2rooms, 3rooms, 4rooms
```

**Текущий snapshot:**

```text
1400 raw rows -> 1359 normalized rows -> 1300 clean rows
```

Очистка включает:

- удаление дублей по `listing_id`/`url`;
- фильтры по цене, площади, rooms_count;
- фильтр broken rows через whitelist 18 официальных районов СПб;
- сохранение студий как `rooms_count = 0`.

**EDA summary:**

| Метрика | Значение |
|---|---:|
| Clean rows | 1300 |
| Median price | 17 700 000 RUB |
| Mean price | 36 466 864 RUB |
| Median area | 58.0 m2 |
| Median price per m2 | 339 430 RUB |
| Median distance to center | 6.78 km |
| Median straight-line distance to metro | 1.41 km |
| Metro route distance coverage | 1253 / 1300 |
| Median walking route distance to metro | 1.76 km |
| Median walking route duration to metro | 21.1 min |

**Геокодинг и метро:**

- `lat`, `lon` получаются через Nominatim по адресу объявления;
- если точный дом не найден, используется street-level fallback;
- если street не найден, используется centroid района;
- `underground` приходит из CIAN как название станции метро;
- координаты метро хранятся в `data/reference/metro_spb_coords.json`;
- `distance_to_metro_km` считается через haversine;
- `distance_to_metro_route_km` и `duration_to_metro_route_min` считаются через openrouteservice `foot-walking` при наличии `OPENROUTESERVICE_API_KEY`.

**Покрытие геокодинга:**

| Гео-уровень | Количество |
|---|---:|
| house | 1016 |
| street | 118 |
| district fallback | 166 |
| total geocoded | 1300 / 1300 |
| metro known | 1253 / 1300 |
| metro route distance | 1253 / 1300 |

**EDA artifacts:**

- `data/processed/cian_eda_summary.md`;
- `data/processed/figures/*.png`;
- price/area/rooms distributions;
- missing values;
- correlation matrix;
- map plot with price per m2;
- distance-to-center and distance-to-metro plots.

### 3. Анализ возможных утечек данных

Не используются как признаки:

- `price`, `log_price`;
- `target_price_per_sqm`, `log_target_price_per_sqm`;
- `observed_price_per_sqm`, `price_per_sqm_eda`;
- `listing_id`, `url`, `source`, `collected_at`;
- `street`, `house_number` как текстовые признаки.

Адресные поля используются только для получения координат. Модель получает численные производные:

```text
lat, lon, distance_to_center_km, distance_to_metro_km,
distance_to_metro_route_km, duration_to_metro_route_min
```

Market aggregates с медианами цены за м2 помечены как leakage risk medium: для честных экспериментов они должны считаться только на train split. В текущем Ridge-эксперименте target-derived aggregates исключаются из train features.

### 4. Data Contract

Контракт реализован в `src/data/contract_cian.py` и документирован в `docs/ML_System_Design_Doc.md`.

Проверка:

```bash
python -m src.data.contract_cian data/processed/cian_spb_clean_geo.csv
```

Результат:

```text
Data Contract: OK
```

Ключевые поля контракта:

| Поле | Required | Диапазон / значения | Max missing | Freshness |
|---|---:|---|---:|---|
| `listing_id` | yes | unique id | 1% | weekly snapshot |
| `url` | yes | valid URL | 0% | weekly snapshot |
| `source` | yes | `cian` | 0% | weekly snapshot |
| `collected_at` | yes | timestamp | 0% | freshness check |
| `price` | yes | 1M-600M RUB | 0% | weekly snapshot |
| `total_meters` | yes | 10-500 m2 | 0% | weekly snapshot |
| `rooms_count` | yes | 0-10 | 2% | weekly snapshot |
| `district` | yes | 18 official SPB districts | 1% | weekly snapshot |
| `lat`, `lon` | optional | SPB bbox | 0% if geocoded | weekly / cache |
| `distance_to_center_km` | optional | 0-50 km | 0% if geocoded | weekly / cache |
| `distance_to_metro_km` | optional | 0-50 km | 5% | weekly / cache |
| `distance_to_metro_route_km` | optional | 0-70 km | up to 100% without API | weekly / route cache |
| `duration_to_metro_route_min` | optional | 0-720 min | up to 100% without API | weekly / route cache |
| `metro_known` | optional | bool | 0% | weekly snapshot |

### 5. Архитектура системы

Архитектура описана в:

- `docs/architecture_cian.md`;
- `docs/architecture_bpmn.md`;
- `docs/ML_System_Design_Doc.md`.

Уровни системы:

1. Data source: CIAN.
2. Extract/Normalize: `collect_cian_spb.py`.
3. Data validation: `contract_cian.py`.
4. Cleaning: `clean_cian.py`.
5. Geo enrichment: Nominatim + metro reference + openrouteservice route API.
6. Storage: CSV snapshots, feature tables, JSON caches.
7. Feature engineering: listing-level features + market aggregates.
8. Modeling: baselines, Ridge, CatBoost-ready training script.
9. Serving layer: FastAPI/Streamlit scaffold for future checkpoints.

Заполненный ML System Design Doc:

```text
docs/ML_System_Design_Doc.md
```

### 6. Риски v0

| Риск | Категория | Причина | Последствие |
|---|---|---|---|
| CIAN parser blocked | Data | anti-bot/layout changes | no fresh data |
| Stale snapshot | Data/Ops | refresh not run | outdated predictions |
| Duplicates/reposts | Data | same listing in several segments | biased EDA/model |
| Broken parser rows | Data | parser puts title into district | bad geocoding/features |
| Luxury outliers | Model | premium listings dominate loss | unstable metrics |
| Target leakage | Model | price-derived features | inflated validation |
| Address memorization | Model | street/house/high-cardinality fields | poor generalization |
| Routing API quota | Infrastructure | openrouteservice limits | route features partly missing |
| Market drift | Operation | market changes over time | degraded predictions |

---

## Чекпоинт 2. Data Engineering и пайплайн данных

### 1. Data Pipeline

**Подход:** ETL.

Причины выбора ETL:

- CIAN - внешний и нестабильный источник;
- raw parser output может содержать дубли, пропуски, outliers и schema drift;
- model-ready storage должен получать уже валидированные и очищенные данные;
- для MVP file-based ETL проще, дешевле и воспроизводимее.

**Этапы пайплайна:**

| Этап | Вход | Выход | Код |
|---|---|---|---|
| Extract | CIAN pages | raw CSV | `src/data/collect_cian_spb.py` |
| Normalize | parser output | normalized CSV | `src/data/collect_cian_spb.py` |
| Validate | clean/geocoded CSV | contract status | `src/data/contract_cian.py` |
| Clean | normalized CSV | clean CSV | `src/data/clean_cian.py` |
| Geocode | clean CSV | geo-enriched CSV | `src/features/geocoder.py` |
| Route distance | house lat/lon + metro lat/lon | walking route distance/duration | openrouteservice API |
| Feature Engineering | geocoded CSV | offline feature table + aggregates | `src/features/build_features.py` |
| Sampling | clean CSV | balanced sample | `src/data/sampling.py` |
| EDA/Reports | geocoded CSV | figures + markdown reports | `src/data/make_cian_eda.py` |

**Запуск:**

```bash
python -m src.pipeline.run_data_pipeline
```

С реальной пешей route-distance до метро:

```bash
python -m src.pipeline.run_data_pipeline --with-routing
```

Ключ openrouteservice хранится локально в `.env`:

```text
OPENROUTESERVICE_API_KEY=...
```

**Актуальные artifacts:**

| Artifact | Shape |
|---|---:|
| `data/processed/cian_spb_clean.csv` | 1300 x 27 |
| `data/processed/cian_spb_clean_geo.csv` | 1300 x 36 |
| `data/features/cian_spb_offline_features.csv` | 1300 x 50 |
| `data/features/cian_spb_district_market_aggregates.csv` | 18 x 6 |
| `data/features/cian_spb_district_rooms_market_aggregates.csv` | 89 x 7 |
| `data/features/cian_spb_underground_market_aggregates.csv` | 73 x 6 |
| `data/processed/cian_spb_balanced_sample.csv` | 1275 x 27 |

### 2. Feature Engineering

**Listing-level признаки:**

| Feature | Source | Refresh | Offline | Online |
|---|---|---|---:|---:|
| `author_type` | CIAN listing | weekly | yes | yes |
| `room_segment` | collector segment | weekly | yes | yes |
| `rooms_count` | CIAN listing | weekly | yes | yes |
| `total_meters` | CIAN listing | weekly/request | yes | yes |
| `log_total_meters` | derived | weekly/request | yes | yes |
| `floor`, `floors_count` | CIAN listing | weekly/request | yes | yes |
| `floor_ratio` | derived | weekly/request | yes | yes |
| `is_first_floor`, `is_last_floor` | derived | weekly/request | yes | yes |
| `district` | CIAN listing + whitelist | weekly/request | yes | yes |
| `underground` | CIAN listing | weekly/request | yes | yes |
| `lat`, `lon` | Nominatim geocoder | weekly/cache | yes | yes |
| `geo_precision` | geocoder tier | weekly/cache | yes | yes |
| `distance_to_center_km` | haversine | weekly/cache | yes | yes |
| `distance_to_metro_km` | haversine | weekly/cache | yes | yes |
| `distance_to_metro_route_km` | openrouteservice foot-walking | weekly/cache/API | yes | yes |
| `duration_to_metro_route_min` | openrouteservice foot-walking | weekly/cache/API | yes | yes |
| `metro_known` | derived flag | weekly/cache | yes | yes |

**Market aggregate features:**

| Feature | Aggregation |
|---|---|
| `district_ads_count` | count by district |
| `district_median_price_per_sqm` | median by district |
| `district_rooms_ads_count` | count by district + rooms |
| `district_rooms_median_price_per_sqm` | median by district + rooms |
| `underground_median_price_per_sqm` | median by metro |
| `room_segment_median_price_per_sqm` | median by room segment |
| `rooms_median_price_per_sqm` | median by rooms |

**Feature Store concept:**

Offline store:

```text
data/features/cian_spb_offline_features.csv
```

Online lookup tables:

```text
data/features/cian_spb_*_market_aggregates.csv
```

Registry:

```text
docs/feature_registry.md
data/features/feature_registry.json
```

Разделение:

- offline features используются для training/validation;
- online lookup tables нужны для будущего API;
- route cache хранится отдельно в `data/cache/route_cache.json`;
- API key не хранится в коде, только в локальном `.env`.

### 3. Работа с данными

Использованный подход: **семплирование / imbalance correction**.

Проблема: разные room-сегменты представлены неравномерно.

Решение:

```text
stratified downsampling by rooms_count
```

Результат:

| Rooms | Rows |
|---:|---:|
| 0 | 255 |
| 1 | 255 |
| 2 | 255 |
| 3 | 255 |
| 4 | 255 |

Artifact:

```text
data/processed/cian_spb_balanced_sample.csv
data/processed/sampling_report.md
```

### 4. DFD

DFD описан в:

```text
docs/dfd_checkpoint2.md
```

Поток данных:

```text
CIAN -> Extract -> Raw snapshots -> Normalize -> Validate -> Clean
     -> Geocode -> Route API/cache -> Feature Engineering
     -> Offline Feature Store + Online Lookup Tables
     -> Sampling + Reports
```

Внешние сущности:

- CIAN;
- OSM Nominatim;
- openrouteservice Directions API.

Data stores:

- raw/normalized snapshots;
- clean dataset;
- geocoded dataset;
- geocode cache;
- route cache;
- offline feature store;
- online lookup tables;
- balanced dataset.

---

## Моделирование как подготовка к Checkpoint 3

Файл:

```text
src/models/train.py
```

Текущий режим:

- train/val/test = 60/20/20;
- stratify by `rooms_count`;
- target = `log_target_price_per_sqm`;
- aggregate leakage columns исключаются;
- Ridge работает стабильно;
- CatBoost поддержан в коде, но может быть пропущен через `--skip-catboost`, если библиотека не установлена.

Результаты текущего Ridge-прогона:

| Модель | R2 на price_per_sqm | MAE, RUB/m2 | MAPE |
|---|---:|---:|---:|
| B2 non-ML baseline | 0.3137 | - | - |
| Ridge | 0.6266 | 96 353 | 21.8% |

Вывод: Ridge уже дает примерно 2x улучшение по R2 относительно лучшего non-ML baseline. Для Checkpoint 3 следующий шаг - оформить полноценный экспериментальный контур: CatBoost/другая модель, experiment tracking, сравнение моделей, сохранение артефактов, анализ feature importance и стратегия retraining.

---

## Команды для проверки

```bash
python -m src.data.contract_cian data/processed/cian_spb_clean_geo.csv
python -m src.pipeline.run_data_pipeline --with-routing
python -m src.models.baseline_cian
python -m src.models.train --skip-catboost
```

Ожидаемые признаки успешного прогона:

```text
Data Contract: OK
Pipeline completed
metro_known=1253/1300
routed_metro=1253/1300
offline feature table: 1300 x 50
Ridge R2 on price_per_sqm ~= 0.6266
```
