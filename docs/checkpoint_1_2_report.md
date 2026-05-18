# Отчёт по чекпоинтам 1 и 2 — CIAN Real Estate Price Intelligence SPB

**Студент:** курзо
**Дата:** 13 мая 2026
**Тема:** ML-система оценки цены за квадратный метр квартир Санкт-Петербурга по объявлениям CIAN

---

## Чекпоинт 1 — Постановка задачи и первичное проектирование

### 1. Выбор темы и постановка задачи

**Тема:** прогнозирование цены за квадратный метр квартиры в Санкт-Петербурге на основе свежих объявлений CIAN.

**Тип ML-задачи:** supervised regression.  
**Таргет:** `log_target_price_per_sqm = log1p(price / total_meters)`.  
**Объект:** объявление о продаже квартиры.  
**Контекст:** публичное веб-приложение, куда пользователь вводит параметры квартиры и получает оценку цены за м².

**Ожидаемый эффект:**
- быстрая оценка рыночной стоимости квартиры;
- прозрачное сравнение с аналогичными объявлениями;
- выявление аномально дорогих или дешёвых вариантов.

**Ограничения:**
- Данные: свежие snapshot'ы CIAN, без Kaggle;
- География: Санкт-Петербург (MVP);
- Бюджет: бесплатный (локальная разработка, бесплатный хостинг);
- Время ответа API: < 500 мс;
- Нагрузка: < 10 RPS (демо).

### 2. Сбор данных и EDA

**Источник:** CIAN (cian.ru), сбор через `cianparser` по 5 сегментам комнатности (studio, 1, 2, 3, 4 комнаты).

**Текущий снимок:** 1300 объявлений после очистки (1400 raw → 1359 normalized → 1300 clean с фильтром broken-rows по 18 официальным районам СПб).

**Основные цифры EDA:**
- Медианная цена: 17 700 000 ₽
- Медианная площадь: 58.0 м²
- Медианная цена за м²: 339 430 ₽/м²
- Медианное расстояние до центра (Дворцовая): 6.78 км
- Медианное расстояние до метро: 1.41 км
- Покрытие геокодинга: 100% (house 78.2%, street 9.1%, district fallback 12.8%)
- Покрытие метро: 96.4% (1253 из 1300)

Сгенерировано 14 графиков EDA: распределения цены/площади/комнат, корреляции, карта СПб с ценой за м², расстояние до центра/метро.

### 3. Data Contract

Контракт данных включает 23 поля. Для каждого указаны: обязательность, диапазон значений, максимальная доля пропусков, требования к свежести. Документирован в [ML_System_Design_Doc.md §4](ML_System_Design_Doc.md).

Ключевые гарантии:
- `district` ∈ один из 18 официальных районов СПб (whitelist)
- `lat`, `lon` — 100% покрытие (трёхуровневый fallback геокодера)
- `distance_to_metro_km` — 96.4% покрытие (остальное — `metro_known = False`)
- `price` — не используется как фича (только для расчёта таргета и метрик)

### 4. Архитектура системы

Документирована в [architecture_bpmn.md](architecture_bpmn.md) и [architecture_cian.md](architecture_cian.md).

6 слоёв:
1. **User Layer** — Streamlit веб-интерфейс
2. **API Layer** — FastAPI `/predict`, `/health`
3. **ML Layer** — CatBoost (основная) + Ridge (baseline) + B0-B5
4. **Feature Layer** — геокодер (Nominatim + кэш), lookup-таблицы
5. **Data Layer** — offline feature store (.csv) + reference data (координаты метро/районов)
6. **Pipeline Layer** — еженедельный ETL

Три диаграммы: BPMN-пайплайн, Sequence (inference flow), Layers (компонентная).

### 5. ML System Design Doc

Заполненный документ: [ML_System_Design_Doc.md](ML_System_Design_Doc.md). Содержит все обязательные разделы: Goals, Constraints, Data Source, Data Contract, Baselines, Leakage Analysis, Architecture, Risks, Checkpoint Roadmap.

### 6. Риски

9 рисков по 4 категориям (данные, модель, инфраструктура, эксплуатация). Для каждого: причина → последствие → mitigation. Основные: блокировка парсера, устаревание данных, luxury-выбросы, target leakage, дрейф распределений.

---

## Чекпоинт 2 — Data Engineering и пайплайн данных

### 1. Data Pipeline (ETL)

**Выбор: ETL** (а не ELT). Обоснование: CIAN — внешний нестабильный источник; сырые данные требуют валидации до попадания в model-ready storage; проект мал, file-based ETL проще и воспроизводимее.

**Этапы:**
1. Extract — сбор страниц CIAN через cianparser → raw CSV
2. Normalize — приведение к стабильной схеме (24 колонки)
3. Validate — data contract через Great Expectations / contract_cian.py
4. Clean — фильтр broken-rows (whitelist районов), outlier-фильтры
5. **Geocode** — Nominatim 3-tier (house/street/district) + JSON-кэш
6. Feature Engineering — offline feature table + market aggregate lookups
7. Sampling — stratified balanced sample
8. Train — Ridge + CatBoost

**Инструменты:** pandas (ETL), geopy (геокодинг), scikit-learn + CatBoost (модели), CSV + JSON (storage).

### 2. Feature Engineering

**17 признаков** в финальной матрице (18 без residential_complex). Полный registry: [feature_registry.md](feature_registry.md).

**Listing-level фичи:** room_segment, rooms_count, total_meters, log_total_meters, floor_ratio, is_first_floor, is_last_floor, district, underground, lat, lon, geo_precision, distance_to_center_km, distance_to_metro_km, metro_known, author_type.

**Market aggregates (online lookup):** 5 таблиц — по district, district+rooms, underground, room_segment, rooms. Содержат медиану, p25, p75 цены за м². **Не используются при обучении** (leakage risk medium) — только для online inference.

**Feature Store:** файловый — `offline_features.csv` (тренировка) + `*_market_aggregates.csv` (serving). Registry в JSON и Markdown.

### 3. Работа с данными

**Подход: семплирование.** Stratified downsampling по `rooms_count` — каждый сегмент уравнен до размера наименьшего (255 строк × 5 сегментов = 1275 строк). Устраняет доминирование 1-комнатных в снапшоте.

### 4. Data Flow Diagram

Документирован в [dfd_checkpoint2.md](dfd_checkpoint2.md). Mermaid-диаграмма: 7 процессов, 7 хранилищ данных, 2 внешние сущности (CIAN, Nominatim), 3 отчёта (validation, feature registry, sampling).

---

## Результаты моделирования (стык чекпоинтов 2 и 3)

| Модель | R² на price_per_sqm | MAE (RUB/м²) | MAPE |
|---|---|---|---|
| B2 (лучший не-ML baseline) | 0.31 | — | — |
| Ridge (линейный) | 0.63 | 96 864 | 21.8% |
| **CatBoost** | **0.71** | **82 809** | **18.6%** |

CatBoost улучшает B2 в 2.3 раза по R². Топ фич: district (24.7%), author_type (13.7%), distance_to_center_km (7.1%), floors_count (6.5%), total_meters (6.1%). Гео-фичи суммарно дают ~22% важности.

---

## Артефакты чекпоинтов 1 и 2

| Требование | Артефакт | Статус |
|---|---|---|
| Тема и постановка | [ML_System_Design_Doc.md §1-2](docs/ML_System_Design_Doc.md) | ✅ |
| Baseline | [baseline_cian.py](src/models/baseline_cian.py) — B0..B5 | ✅ |
| EDA | [cian_eda_summary.md](data/processed/cian_eda_summary.md) + 14 figs | ✅ |
| Data Contract | [ML_System_Design_Doc.md §4](docs/ML_System_Design_Doc.md) — 23 поля | ✅ |
| Архитектура | [architecture_bpmn.md](docs/architecture_bpmn.md) — 3 диаграммы | ✅ |
| ML System Design Doc | [ML_System_Design_Doc.md](docs/ML_System_Design_Doc.md) | ✅ |
| Риски | [ML_System_Design_Doc.md §8](docs/ML_System_Design_Doc.md) — 9 рисков | ✅ |
| ETL Pipeline | [checkpoint2_data_engineering.md §1](docs/checkpoint2_data_engineering.md) | ✅ |
| Feature Engineering | [feature_registry.md](docs/feature_registry.md) — 17 фич + registry | ✅ |
| Семплирование | [sampling_report.md](data/processed/sampling_report.md) — balanced 1275 rows | ✅ |
| DFD | [dfd_checkpoint2.md](docs/dfd_checkpoint2.md) — Mermaid диаграмма | ✅ |
| ML-модели (сверх программы) | [train.py](src/models/train.py) — Ridge + CatBoost | ✅ |

---

## Запуск пайплайна

```bash
# Полный пайплайн (на существующем снапшоте)
python -m src.pipeline.run_data_pipeline

# Со сбором свежего снапшота
python -m src.pipeline.run_data_pipeline --collect --pages 10 --timeout 20

# Baselines
python -m src.models.baseline_cian

# ML-модели
python -m src.models.train
```
