"""FastAPI сервис для предсказания цен на недвижимость."""

import logging
import time

from fastapi import FastAPI
from pydantic import BaseModel, Field

from src.api.predictor import predict_price

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger(__name__)

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
    distance_to_metro_route_km: float | None = None
    features: dict | None = None
    kafka_event_published: bool = False


@app.get("/health")
def health():
    return {"status": "ok", "version": "0.1.0"}


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    started = time.perf_counter()
    result = predict_price(**request.dict())
    http_latency_ms = (time.perf_counter() - started) * 1000

    logger.info(
        "http predict: model=%s district=%s rooms=%d area=%.0f price=%.0f http_latency_ms=%.1f kafka=%s",
        result["model_name"],
        request.district,
        request.rooms_count,
        request.total_meters,
        result["predicted_price"],
        http_latency_ms,
        result.get("kafka_event_published", False),
    )

    return PredictionResponse(
        predicted_price=result["predicted_price"],
        predicted_price_per_sqm=result["predicted_price_per_sqm"],
        price_range_low=result["price_range_low"],
        price_range_high=result["price_range_high"],
        model_name=result["model_name"],
        geo_precision=result["features"].get("geo_precision"),
        distance_to_center_km=result["features"].get("distance_to_center_km"),
        distance_to_metro_route_km=result["features"].get("distance_to_metro_route_km"),
        features=result["features"],
        kafka_event_published=result.get("kafka_event_published", False),
    )
