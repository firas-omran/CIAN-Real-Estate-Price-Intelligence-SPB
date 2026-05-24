---
name: src/data
description: Сбор, нормализация, валидация контракта, очистка и EDA свежих CIAN-объявлений по СПб. Точка входа сырых данных в проект.
status: done
owner: kurzo
---

# Модуль `src/data` — сбор и подготовка CIAN-данных

## Назначение

Один абзац: модуль отвечает за всё, что превращает «CIAN на сайте» в
проверенный clean-датасет на диске. Парсит CIAN через `cianparser`,
нормализует в стабильную схему проекта, валидирует Data Contract,
чистит выбросы и сломанные строки, считает EDA-сводку. Ниже по
пайплайну (`src/features/`) принимает уже проверенный `cian_spb_clean.csv`.

## Контракт

**Вход:** обращения к `cian.ru` через `cianparser==1.0.4`. Локализация —
`Санкт-Петербург`, deal_type `sale`, сегменты по `rooms_count`
(`studio, 1, 2, 3, 4`).

**Выход:**

| Файл | Содержимое |
|---|---|
| `data/raw/cian_spb_raw_*.csv` | сырые ряды парсера со всеми колонками cianparser (gitignored) |
| `data/raw/cian_spb_normalized_*.csv` | нормализованная схема проекта (NORMALIZED_COLUMNS) |
| `data/processed/cian_spb_clean.csv` | очищенный датасет, готовый для feature engineering и EDA |
| `data/processed/cian_eda_summary.md` | человекочитаемый EDA-отчёт |
| `data/processed/figures/*.png` | EDA-графики |
| `data/processed/cian_spb_balanced_sample.csv` | стратифицированный сэмпл по `rooms_count` |
| `data/processed/sampling_report.md` | отчёт о сэмплировании |

**Ключевые гарантии clean-датасета** (после `clean_cian.py`):
- `price ∈ [1_000_000; 600_000_000]` RUB
- `total_meters ∈ [10; 500]` m²
- `rooms_count ∈ [0; 10]` (0 — студия)
- `price_per_sqm_eda ∈ [50_000; 3_000_000]` RUB/m²
- `district ∈ VALID_SPB_DISTRICTS` (18 официальных районов СПб)
- удалены дубликаты по `listing_id`/`url`
- текстовые поля trim'нуты, пропуски заменены на `"unknown"`

`VALID_SPB_DISTRICTS` экспортируется как `frozenset` из
`src/data/clean_cian.py` и переиспользуется EDA, geocoder и
feature builder — это **единственная** канонизация районов в проекте.

## Зависимости

**Откуда получаем:** `cianparser` (PyPI), интернет до `cian.ru`.

**Кому передаём:**
- `src/features/build_features.py` читает `cian_spb_clean.csv`;
- `src/features/geocoder.py` читает `cian_spb_clean.csv` (district, street, house_number, underground);
- `src/models/baseline_cian.py` читает `cian_spb_clean.csv`;
- `src/pipeline/run_data_pipeline.py` оркестрирует все шаги.

**Внешние библиотеки:** `cianparser`, `pandas`, `numpy`,
`matplotlib`, `seaborn` (для EDA).

## Внутреннее устройство

| Файл | Что делает |
|---|---|
| `collect_cian.py` | низкоуровневый сбор одного сегмента через `cianparser.CianParser.get_flats` |
| `collect_cian_spb.py` | оркестратор сбора по 5 room-сегментам с дедупом, нормализация в `NORMALIZED_COLUMNS` |
| `contract_cian.py` | проверка нормализованного CSV против data contract |
| `clean_cian.py` | бизнес-фильтры по диапазонам, добавление `log_price`, `floor_ratio`, фильтр broken-rows по `VALID_SPB_DISTRICTS` |
| `make_cian_eda.py` | человекочитаемый EDA: статистики, графики, отчёт |
| `sampling.py` | стратифицированный сэмпл по `rooms_count` для balanced-эксперимента |

CLI: каждый файл запускается через `python -m src.data.<module>`.

## Тесты

Юнит-тестов сейчас нет (TODO ниже). Вместо них используется
прогон полного пайплайна `python -m src.pipeline.run_data_pipeline`
с фикс-сидом на чистом снапшоте; результат сравнивается визуально с
`docs/checkpoint1_cian_report.md` секцией 5 и
`docs/checkpoint2_data_engineering.md` секцией 1.

Команда полного прогона:

```bash
python -m src.pipeline.run_data_pipeline --collect --pages 10 --timeout 20
```

Без `--collect` пайплайн использует существующий снапшот.

## TODO

- [x] Очистка broken-rows по `VALID_SPB_DISTRICTS` (2026-05-06).
- [x] EDA-расширение: секция Geocoding Coverage и 4 новые фигуры (`distance_to_center_distribution.png`, `distance_to_metro_distribution.png`, `price_per_sqm_vs_distance.png`, `spb_map_price_per_sqm.png`) (2026-05-06).
- [ ] Юнит-тесты на ключевые фильтры (`clean_cian_frame`, `normalize_cian_frame`).
- [ ] Расширение парсера через `--with-extra-data` (год постройки,
  тип дома, площадь кухни) — отдельной итерацией, см. §13 спеки.

## Документация и ссылки

- ML System Design Doc: `docs/ML_System_Design_Doc.md`
- Data Engineering report: `docs/checkpoint2_data_engineering.md`
- Активная спека текущей итерации: `docs/superpowers/specs/2026-05-06-target-switch-and-geo-design.md`
- Парсер CIAN: <https://github.com/lenarsaitov/cianparser>
