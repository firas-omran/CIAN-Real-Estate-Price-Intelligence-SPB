"""FastAPI сервис для предсказания цен на недвижимость."""

import time

from fastapi import FastAPI
from pydantic import BaseModel, Field

from src.api.predictor import predict_price
from src.api.telemetry import PREDICTION_LATENCY, publish_prediction_event, record_prediction_metrics

try:
    from prometheus_client import make_asgi_app
except ModuleNotFoundError:  # pragma: no cover - optional production dependency
    make_asgi_app = None

app = FastAPI(
    title="CIAN Real Estate Price Intelligence API",
    description="API для оценки стоимости квартиры в Санкт-Петербурге по свежим данным CIAN",
    version="0.1.0",
)
if make_asgi_app is not None:
    app.mount("/metrics", make_asgi_app())


class PredictionRequest(BaseModel):
    rooms_count: int = Field(ge=0, le=10, description="0 means studio")
    total_meters: float = Field(gt=10, le=500)
    floor: int = Field(ge=1, le=100)
    floors_count: int = Field(ge=1, le=100)
    district: str
    underground: str = "unknown"
    author_type: str = "real_estate_agent"
    street: str | None = None
    house_number: str | None = None


class PredictionResponse(BaseModel):
    predicted_price: float
    predicted_price_per_sqm: float
    price_range_low: float
    price_range_high: float
    model_name: str
    geo_precision: str | None
    distance_to_center_km: float | None
    kafka_event_published: bool = False


@app.get("/health")
def health():
    return {"status": "ok", "version": "0.1.0"}


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    started = time.perf_counter()
    result = predict_price(**request.dict())
    latency_seconds = time.perf_counter() - started
    if PREDICTION_LATENCY is not None:
        PREDICTION_LATENCY.observe(latency_seconds)

    record_prediction_metrics(result["model_name"], result["predicted_price"], status="ok")
    kafka_event_published = publish_prediction_event(
        {
            "model_name": result["model_name"],
            "predicted_price": result["predicted_price"],
            "predicted_price_per_sqm": result["predicted_price_per_sqm"],
            "district": request.district,
            "rooms_count": request.rooms_count,
            "total_meters": request.total_meters,
            "latency_ms": round(latency_seconds * 1000, 2),
        }
    )
    return PredictionResponse(
        predicted_price=result["predicted_price"],
        predicted_price_per_sqm=result["predicted_price_per_sqm"],
        price_range_low=result["price_range_low"],
        price_range_high=result["price_range_high"],
        model_name=result["model_name"],
        geo_precision=result["features"].get("geo_precision"),
        distance_to_center_km=result["features"].get("distance_to_center_km"),
        kafka_event_published=kafka_event_published,
    )
