---
name: src/models
description: Non-ML baselines (B0..B3) для прогноза цены квартир СПб; подготовка площадки под ML-модели чекпоинта 3.
status: done
owner: kurzo
---

# Модуль `src/models` — baselines и (в будущем) ML-модели

## Назначение

Считает референсные не-ML baselines, которые задают планку «без модели»
для будущих ML-экспериментов. Текущая итерация переводит baselines на
новый таргет `price_per_sqm` (обучаются на per-sqm, восстанавливают
`price = pred × total_meters` для бизнес-метрик).

ML-модели (CatBoost, Linear, Quantile-CatBoost) будут добавлены в
чекпоинте 3 — сейчас в `src/models/` живут только baselines.

## Контракт

**Вход:** `data/processed/cian_spb_clean.csv` (после фильтра broken-rows
из `src/data/clean_cian.py`) — обязательные колонки: `price, total_meters,
rooms_count, district, price_per_sqm_eda`.

**Выход:**

| Файл | Содержимое |
|---|---|
| `data/processed/baseline_metrics.csv` | таблица B0..B3 с метриками `MAE` (RUB), `MAPE` (%), `R²` на price, `R²` на price_per_sqm |

**Метрики:**
- Бизнес-метрики считаются на восстановленной `price` для сопоставимости с
  ML System Design Doc §5.
- `R²` на `price_per_sqm` — отдельная «честная» метрика: показывает,
  понимает ли модель рынок без вклада площади.

**Воспроизводимый split:** 80/20 по `random_state=42`,
`pd.DataFrame.sample(frac=1.0)` — соответствует `train_test_split` в
`baseline_cian.py:23-27`.

## Зависимости

**От `src/data/`:** clean-датасет.

**От `src/features/`** (после миграции на новый таргет): offline feature
table будет содержать `target_price_per_sqm` и market aggregates,
которые B2/B3 могут переиспользовать вместо повторного агрегирования.

**Внешние:** `pandas`, `numpy`. Не использует ML-фреймворков.

## Внутреннее устройство

| Файл | Что делает |
|---|---|
| `baseline_cian.py` | четыре бейзлайна B0..B3 + helper'ы для метрик |

| Бейзлайн | Логика |
|---|---|
| **B0** | глобальная медиана `price_per_sqm` |
| **B1** | медиана `price_per_sqm` по `rooms_count` |
| **B2** | медиана `price_per_sqm` по `(district, rooms_count)` |
| **B3** | KNN-стиль: внутри `(district, rooms_count)` берём `k=10` ближайших по `total_meters`, медиана их `price_per_sqm` |

После предсказания `price_per_sqm_pred` восстанавливаем
`price_pred = price_per_sqm_pred × total_meters` для расчёта бизнес-метрик.

## Тесты

- Smoke-тест: на маленьком фрейме (10-20 строк) все 4 бейзлайна возвращают
  ненулевые предсказания и метрики не NaN.
- Integration: после прогона на полном clean-датасете метрики сохраняются
  и матчатся с цифрами в `docs/checkpoint1_cian_report.md` §5.

Команда:

```bash
python -m src.models.baseline_cian
```

## TODO

- [x] Переписать B0..B3 на новый таргет `price_per_sqm` (2026-05-06).
- [x] Добавить метрику `R²` на price и `R²` на price_per_sqm в `MetricResult` (2026-05-06).
- [ ] Чекпоинт 3: Linear, CatBoost, Quantile-CatBoost — отдельный модуль
  (например `src/models/ml_models.py`).
- [ ] Чекпоинт 4: prediction intervals + светофор «рыночная цена».

## Документация и ссылки

- Активная спека: `docs/superpowers/specs/2026-05-06-target-switch-and-geo-design.md` §8 (Baselines Update)
- ML System Design Doc §5 (Baselines and Metrics)
- Checkpoint 1 report §5 (исторические числа на старом таргете `price`)
