# CIAN Real Estate Price Intelligence

ML-система для оценки рыночной стоимости квартир в Санкт-Петербурге по свежим объявлениям CIAN.

## Описание

Проект собирает актуальные объявления CIAN, валидирует Data Contract, очищает данные (с фильтром broken-rows по 18 официальным районам СПб), геокодирует адрес каждого объявления через OpenStreetMap Nominatim, строит EDA и baseline-оценки без ML. Активный таргет — `target_price_per_sqm = price / total_meters`; цена восстанавливается умножением на `total_meters` для бизнес-метрик. Kaggle-данные не используются.

## Архитектура

- **Backend:** FastAPI (ML-сервис, REST API)
- **Frontend:** Streamlit (веб-интерфейс)
- **ML:** scikit-learn / CatBoost
- **Данные:** fresh CIAN snapshot, Санкт-Петербург

## Установка

```bash
python -m venv .venv
.venv\Scripts\activate  # Windows
pip install -r requirements.txt
```

## Как собрать данные

```bash
python -m src.data.collect_cian_spb --pages 10 --timeout 20
```

Для более крупного датасета:

```bash
python -m src.data.collect_cian_spb --pages 20 --timeout 20
```

Ожидаемый размер: примерно 1400 объявлений при 10 страницах и 2800 объявлений при 20 страницах до удаления дублей.

Рекомендация для защиты: собрать `--pages 20`, затем запустить полный pipeline ниже. Текущий рабочий snapshot `--pages 10` содержит 1400 raw rows, 1359 normalized rows и 1304 clean rows, включая segment `studio`.

## Pipeline для Checkpoint 1

```bash
python -m src.data.clean_cian
python -m src.data.contract_cian data/processed/cian_spb_clean.csv
python -m src.features.geocoder
python -m src.data.make_cian_eda
python -m src.models.baseline_cian
```

## Pipeline для Checkpoint 2

```bash
python -m src.pipeline.run_data_pipeline
```

Сбор свежего snapshot + полный pipeline (включая геокодинг):

```bash
python -m src.pipeline.run_data_pipeline --collect --pages 10 --timeout 20
```

Шаг геокодинга использует Nominatim (rate limit 1 req/s) и кэширует результаты в `data/cache/geocode_cache.json`, `data/reference/metro_spb_coords.json`, `data/reference/spb_district_centroids.json`. На свежем snapshot первый прогон занимает ~10-15 минут; последующие прогоны мгновенные за счёт кэша.

Результаты:

```text
data/processed/cian_spb_clean.csv
data/processed/cian_spb_clean_geo.csv
data/features/cian_spb_offline_features.csv
data/features/cian_spb_*_market_aggregates.csv
data/processed/cian_spb_balanced_sample.csv
data/processed/pipeline_report.md
data/processed/figures/*.png
data/processed/cian_eda_summary.md
data/processed/baseline_metrics.csv
docs/checkpoint1_cian_report.md
docs/checkpoint2_data_engineering.md
docs/feature_registry.md
docs/dfd_checkpoint2.md
docs/ML_System_Design_Doc.md
docs/architecture_cian.md
docs/superpowers/specs/2026-05-06-target-switch-and-geo-design.md
```

## Структура проекта

```
real-estate-price-explorer/
|-- src/           # Исходный код
|-- notebooks/     # Jupyter notebooks (EDA)
|-- tests/         # Тесты
|-- data/          # Данные (не в git)
|-- docs/          # Документация
```

## Чекпоинты

- [x] Чекпоинт 1: Постановка задачи и первичное проектирование на CIAN
- [x] Чекпоинт 2: Data Engineering и пайплайн данных
- [ ] Чекпоинт 3: Моделирование и эксперименты
- [ ] Чекпоинт 4: Деплой, мониторинг и эксплуатация
