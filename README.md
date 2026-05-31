# CIAN Real Estate Price Intelligence

ML-система для оценки рыночной стоимости квартир в Санкт-Петербурге по свежим объявлениям CIAN.

## Описание

Проект собирает актуальные объявления CIAN, валидирует Data Contract, очищает данные (с фильтром broken-rows по 18 официальным районам СПб), геокодирует адрес каждого объявления через OpenStreetMap Nominatim, строит EDA и baseline-оценки без ML. Гео-блок хранит координаты объявления, прямые haversine-расстояния до центра/метро и опционально настоящую пешую route-distance до метро через openrouteservice. Активный таргет — `target_price_per_sqm = price / total_meters`; цена восстанавливается умножением на `total_meters` для бизнес-метрик. Kaggle-данные не используются.

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

Рекомендация для защиты: собрать `--pages 20`, затем запустить полный pipeline ниже. Текущий рабочий snapshot `--pages 10` содержит 1400 raw rows, 1359 normalized rows и 1300 clean rows после broken-row фильтра по районам СПб, включая segment `studio`.

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

Пешая route-distance до метро через openrouteservice:

```bash
export OPENROUTESERVICE_API_KEY="your_key"
python -m src.pipeline.run_data_pipeline --with-routing
```

Чтобы не вводить ключ каждый раз, создайте локальный `.env`:

```bash
cp .env.example .env
```

Затем откройте `.env` и замените `put_your_key_here` на свой ключ.
Файл `.env` уже добавлен в `.gitignore`.

Если ключ не задан, pipeline всё равно работает: `distance_to_metro_route_km`
и `duration_to_metro_route_min` остаются пустыми, а базовая
`distance_to_metro_km` считается через haversine.

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
data/experiments/checkpoint3_metrics.csv
data/experiments/checkpoint3_metadata.json
docs/checkpoint1_cian_report.md
docs/checkpoint2_data_engineering.md
docs/checkpoint3_modeling.md
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
- [x] Чекпоинт 3: Моделирование и эксперименты
- [x] Чекпоинт 4: Деплой, мониторинг и эксплуатация

## Checkpoint 3 experiments

```bash
python -m src.models.experiments
```

Автоматическая проверка retraining-триггеров и запуск переобучения при необходимости:

```bash
python -m src.pipeline.auto_retrain
```

Проверить решение без запуска переобучения:

```bash
python -m src.pipeline.auto_retrain --dry-run
```

Основной отчёт:

```text
docs/checkpoint3_modeling.md
```

## Checkpoint 4 demo

API:

```bash
arch -arm64 python -m uvicorn src.api.main:app --reload --port 8000
```

Streamlit app:

```bash
arch -arm64 python -m streamlit run src/app/streamlit_app.py
```

Open the Streamlit tab `Monitoring & Drift` and switch from
`Normal snapshot` to `Degradation / drift demo` to show data drift, alerts,
and runbook actions.

Main report:

```text
docs/checkpoint4_deployment_monitoring.md
```

Optional production-like MLOps stack:

```bash
docker compose up --build
```

Services:

```text
FastAPI:    http://localhost:8000/docs
Streamlit:  http://localhost:8501
MLflow:     http://localhost:5001
Kafka UI:   http://localhost:8085
Prometheus: http://localhost:9090
Grafana:    http://localhost:3000  (admin / admin)
Airflow:    http://localhost:8080  (admin / admin)
```
