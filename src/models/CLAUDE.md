---
name: src/models
description: Non-ML baselines (B0..B5) и ML-модели (Ridge, CatBoost) для прогноза price_per_sqm квартир СПб.
status: in_progress
owner: kurzo
---

# Модуль `src/models` — baselines и ML-модели

## Назначение

Считает референсные не-ML baselines (B0..B5) и обучает ML-модели (Ridge, CatBoost)
на таргете `log_target_price_per_sqm`. Все метрики — на шкале `price_per_sqm`
(восстановление через `expm1`). Реконструкция `price` для бизнес-метрик не
выполняется — предсказываем только цену за кв. метр.

## Контракт

**Вход:**
- Baselines: `data/processed/cian_spb_clean_geo.csv`
- ML-модели: `data/features/cian_spb_offline_features.csv` (18 фич, исключены leakage и aggregates)

**Выход:**

| Файл | Содержимое |
|---|---|
| `data/processed/baseline_metrics.csv` | B0..B5 с метриками `MAE`, `MAPE`, `R^2 price`, `R^2 per_sqm` |
| `data/processed/ml_metrics.csv` | Ridge + CatBoost с `R^2 per_sqm`, `MAE per_sqm`, `MAPE` |
| `data/models/catboost_price_per_sqm.cbm` | CatBoost модель |
| `data/models/ridge_price_per_sqm.pkl` | Ridge пайплайн |

**Метрики (test, 60/20/20 split, seed=42):**

| Модель | R^2 per_sqm | MAE (RUB/m^2) | MAPE |
|---|---|---|---|
| B2 (district+rooms) | 0.31 | — | — |
| Ridge | 0.63 | 96,864 | 21.8% |
| **CatBoost** | **0.71** | **82,809** | **18.6%** |

## Зависимости

**От `src/features/`:** `cian_spb_clean_geo.csv` (baselines), `cian_spb_offline_features.csv` (ML).

**Внешние:** `pandas`, `numpy`, `scikit-learn`, `catboost`, `joblib`.

## Внутреннее устройство

| Файл | Что делает |
|---|---|
| `baseline_cian.py` | шесть бейзлайнов B0..B5 + helper'ы для метрик и гео-биннинга |
| `train.py` | загрузка фич, train/val/test split, Ridge baseline, CatBoost, CV, feature importance, сохранение моделей |

### Baselines

| Бейзлайн | Логика |
|---|---|
| **B0** | глобальная медиана `price_per_sqm` |
| **B1** | медиана `price_per_sqm` по `rooms_count` |
| **B2** | медиана `price_per_sqm` по `(district, rooms_count)` |
| **B3** | KNN-стиль: внутри `(district, rooms_count)` берём `k=10` ближайших по `total_meters`, медиана их `price_per_sqm` |
| **B4** | медиана `price_per_sqm` по `(center_distance_bin, rooms_count)` — 5 бакетов: 0-3, 3-6, 6-10, 10-15, 15+ км |
| **B5** | медиана `price_per_sqm` по `(metro_distance_bin, rooms_count)` — 5 бакетов: 0-0.5, 0.5-1, 1-2, 2-5, 5+ км |

### ML Pipeline (train.py)

1. Загрузка `cian_spb_offline_features.csv`
2. Исключение leakage (price, log_price, listing_id, market aggregates, residential_complex)
3. 60/20/20 split с стратификацией по `rooms_count`
4. Ridge: StandardScaler + OneHotEncoder + median imputer
5. CatBoost: нативные категориальные фичи + NaN handling
6. 5-fold CV на train, финальная оценка на test

**Топ-5 фич CatBoost:** district (24.7%), author_type (13.7%), distance_to_center_km (7.1%), floors_count (6.5%), total_meters (6.1%).

## Тесты

- Baselines: smoke-тест на 10-20 строк, все 6 бейзлайнов возвращают ненулевые предсказания.
- ML: `py -m src.models.train` — полный прогон, проверка что R^2 > B2.

Команды:

```bash
python -m src.models.baseline_cian
python -m src.models.train
```

## TODO

- [x] Переписать B0..B3 на новый таргет `price_per_sqm` (2026-05-06).
- [x] Добавить B4 (center_distance_bin) и B5 (metro_distance_bin) — гео-бейзлайны (2026-05-13).
- [x] Ridge baseline + CatBoost модель (2026-05-13). R^2 per_sqm: 0.71 (2.3x от B2).
- [ ] Добавить LightGBM / XGBoost для сравнения.
- [ ] Quantile CatBoost для prediction intervals (чекпоинт 4).
- [ ] Оптимизация гиперпараметров (Optuna / GridSearchCV).
- [ ] Сохранение one-hot encoder / feature names для inference API.

## Документация и ссылки

- Активная спека: `docs/superpowers/specs/2026-05-06-target-switch-and-geo-design.md`
- ML System Design Doc: `docs/ML_System_Design_Doc.md` §5
- Data Engineering report: `docs/checkpoint2_data_engineering.md`
