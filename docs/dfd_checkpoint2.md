# DFD - Checkpoint 2 Data Flow

```mermaid
flowchart LR
    UserOps[Project Team] -->|run pipeline| Pipeline[Pipeline Orchestrator]

    CIAN[External Entity: CIAN] -->|listing pages| P1[Process 1: Extract Listings]
    P1 -->|raw parser output| D1[(Data Store: Raw Snapshots)]

    D1 --> P2[Process 2: Normalize Schema]
    P2 -->|normalized listings| D2[(Data Store: Normalized Snapshots)]

    D2 --> P3[Process 3: Validate Data Contract]
    P3 -->|validation status| R1[Validation Report]
    P3 -->|valid data| P4[Process 4: Clean Data]

    P4 -->|clean listings| D3[(Data Store: Clean Dataset)]
    D3 --> P5[Process 5: Feature Engineering]

    P5 -->|listing-level features + target| D4[(Offline Feature Store)]
    P5 -->|market aggregate lookups| D5[(Online Feature Lookup Tables)]
    P5 -->|feature metadata| R2[Feature Registry]

    D3 --> P6[Process 6: Stratified Sampling]
    P6 -->|balanced sample| D6[(Balanced Dataset)]
    P6 -->|sampling summary| R3[Sampling Report]

    D4 --> FutureTrain[Future Process: Model Training]
    D5 --> FutureAPI[Future Process: Prediction API]
```

## Data Stores

| Store | Description |
|---|---|
| Raw Snapshots | unmodified CIAN parser outputs |
| Normalized Snapshots | stable project schema with selected fields |
| Clean Dataset | validated and cleaned model-ready listing data |
| Offline Feature Store | training feature matrix with targets |
| Online Feature Lookup Tables | aggregate market features for serving |
| Balanced Dataset | room-balanced sample for analysis and experiments |

## Difference From Architecture Diagram

The architecture diagram shows system components and future serving/training
blocks. This DFD focuses specifically on how data moves between processes and
stores in Checkpoint 2.
