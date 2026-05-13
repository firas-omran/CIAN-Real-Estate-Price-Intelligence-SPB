# CIAN Price Intelligence — Architecture Diagrams

## 1. Inference Flow (Sequence Diagram)

Sequence diagram shows the runtime request path when a user requests a price estimate.

```mermaid
sequenceDiagram
    autonumber
    actor User as Пользователь
    participant API as Prediction API<br/>(FastAPI)
    participant GEO as Geocoder<br/>(геокодер)
    participant NOM as Nominatim<br/>(OSM)
    participant FS as Feature Store<br/>(lookup tables)
    participant ML as ML Model<br/>(CatBoost)

    User->>API: POST /predict<br/>{ district, street, house, rooms, total_meters, floor, floors_count, underground }

    Note over API: Валидация входных параметров

    API->>GEO: enrich_listing(district, street, house, underground)

    GEO->>GEO: cache hit?
    alt cache miss
        GEO->>NOM: structured query { street, city }
        NOM-->>GEO: lat, lon (precision: house/street)
        GEO->>GEO: записать в geocode_cache.json
    end
    GEO->>GEO: haversine(lat, lon, CENTER) → distance_to_center_km
    GEO->>GEO: haversine(lat, lon, metro_coords) → distance_to_metro_km
    GEO-->>API: { lat, lon, geo_precision, distance_to_center_km, distance_to_metro_km, metro_known }

    API->>FS: lookup district_rooms_median_price_per_sqm<br/>(district, rooms_count)
    FS-->>API: market aggregate features

    Note over API: Сборка вектора признаков

    API->>ML: predict(feature_vector)
    ML-->>API: log_price_per_sqm_pred

    Note over API: price_pred = expm1(log_price_per_sqm_pred) × total_meters

    API-->>User: { price_estimate, price_per_sqm, confidence_interval, comparable_market_stats }
```

---

## 2. Data Pipeline (BPMN Swimlane)

BPMN-style swimlane diagram of the weekly ETL pipeline.

```mermaid
flowchart TB
    subgraph EXT ["External Systems"]
        CIAN["fa:fa-globe CIAN\n(cian.ru)"]
        NOM2["fa:fa-map Nominatim\n(OSM API)"]
    end

    subgraph COLLECT ["Lane 1: Collection & Normalization"]
        direction TB
        P1["Extract Listings\ncian_spb_collect.py\n───\nIN: CIAN pages\nOUT: raw_*.csv"]
        P2["Normalize Schema\nNORMALIZED_COLUMNS\n───\nIN: raw_*.csv\nOUT: normalized_*.csv"]
        P3["Validate Contract\ncontract_cian.py\n───\nIN: normalized_*.csv\nOUT: validation report"]
    end

    subgraph CLEAN ["Lane 2: Cleaning & Geocoding"]
        direction TB
        P4["Clean Data\nclean_cian.py\n───\nFilter: VALID_SPB_DISTRICTS\nIN: normalized | OUT: clean.csv\n1400→1300 rows"]
        P5["Geocode Listings\ngeocoder.py\n───\nNominatim 3-tier + cache\nIN: clean.csv | OUT: clean_geo.csv\n+6 geo columns"]
    end

    subgraph FEAT ["Lane 3: Feature Engineering"]
        direction TB
        P6["Build Offline Features\nbuild_features.py\n───\ntarget_price_per_sqm\nlog_target_price_per_sqm\nIN: clean_geo.csv | OUT: offline_features.csv"]
        P7["Build Market Aggregates\nbuild_features.py\n───\nmedian price_per_sqm\nby district/rooms/metro\nOUT: *_market_aggregates.csv"]
    end

    subgraph TRAIN ["Lane 4: Training (Checkpoint 3)"]
        direction TB
        P8["Train ML Models\nLinear, CatBoost, Quantile\n───\nIN: offline_features.csv\nOUT: model artifacts + metrics"]
    end

    subgraph SERVE ["Lane 5: Serving"]
        direction TB
        P9["Prediction API\nFastAPI\n───\nIN: user request\nOUT: price estimate"]
    end

    CIAN -->|listing pages| P1
    P1 --> P2
    P2 --> P3
    P3 -->|valid rows| P4
    P4 --> P5
    NOM2 -->|lat, lon| P5
    P5 --> P6
    P5 --> P7
    P6 --> P8
    P7 --> P9
    P8 --> P9

    style EXT fill:#1e3a5f,color:#fff,stroke:#4a90d9
    style COLLECT fill:#1a3d2b,color:#fff,stroke:#4caf50
    style CLEAN fill:#3d2b1a,color:#fff,stroke:#ff9800
    style FEAT fill:#2b1a3d,color:#fff,stroke:#9c27b0
    style TRAIN fill:#3d1a1a,color:#fff,stroke:#f44336
    style SERVE fill:#1a3d3d,color:#fff,stroke:#00bcd4
```

---

## 3. System Architecture Layers

```mermaid
flowchart LR
    subgraph USER ["User Layer"]
        UI["Web UI\n(React / Streamlit)"]
        ANALYST["Analyst\n(Jupyter)"]
    end

    subgraph API_LAYER ["API Layer"]
        FASTAPI["FastAPI\n/predict\n/health"]
    end

    subgraph ML_LAYER ["ML Layer"]
        MODEL["CatBoost\n(Checkpoint 3)"]
        BASELINE["Baselines B0-B3\n(current)"]
    end

    subgraph FEATURE_LAYER ["Feature Layer"]
        GEO_SVC["Geocoder\ngeocoder.py"]
        LOOKUP["Market Aggregates\n(CSV lookup)"]
        CACHE["Geocode Cache\ngeocode_cache.json"]
    end

    subgraph DATA_LAYER ["Data Layer"]
        OFFLINE["Offline Feature Store\ncian_spb_offline_features.csv"]
        REF["Reference Data\nmetro_spb_coords.json\nspb_district_centroids.json"]
    end

    subgraph PIPELINE ["Pipeline (weekly)"]
        ETL["ETL Pipeline\nrun_data_pipeline.py"]
    end

    UI --> FASTAPI
    ANALYST --> OFFLINE

    FASTAPI --> MODEL
    FASTAPI --> BASELINE
    FASTAPI --> GEO_SVC
    FASTAPI --> LOOKUP

    GEO_SVC --> CACHE
    GEO_SVC --> REF

    MODEL --> OFFLINE
    BASELINE --> OFFLINE

    ETL --> OFFLINE
    ETL --> LOOKUP
    ETL --> CACHE

    style USER fill:#0d1b2a,color:#e0e0e0,stroke:#4a90d9
    style API_LAYER fill:#0d2218,color:#e0e0e0,stroke:#4caf50
    style ML_LAYER fill:#1f0d2a,color:#e0e0e0,stroke:#9c27b0
    style FEATURE_LAYER fill:#2a1a0d,color:#e0e0e0,stroke:#ff9800
    style DATA_LAYER fill:#2a0d0d,color:#e0e0e0,stroke:#f44336
    style PIPELINE fill:#0d2a2a,color:#e0e0e0,stroke:#00bcd4
```
