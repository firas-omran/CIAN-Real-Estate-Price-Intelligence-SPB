# CIAN Real Estate Price Intelligence

ML-система для оценки рыночной стоимости квартир в Санкт-Петербурге по свежим объявлениям CIAN.

## Описание

Проект собирает актуальные объявления CIAN, валидирует Data Contract, очищает данные, строит EDA и baseline-оценки без ML. Kaggle-данные больше не используются.

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
python -m src.data.make_cian_eda
python -m src.models.baseline_cian
```

## Pipeline для Checkpoint 2

```bash
python -m src.pipeline.run_data_pipeline
```

Сбор свежего snapshot + полный pipeline:

```bash
python -m src.pipeline.run_data_pipeline --collect --pages 10 --timeout 20
```

Результаты:

```text
data/processed/cian_spb_clean.csv
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
