"""FastAPI сервис для предсказания цен на недвижимость."""

from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(
    title="CIAN Real Estate Price Intelligence API",
    description="API для оценки стоимости квартиры в Санкт-Петербурге по свежим данным CIAN",
    version="0.1.0",
)


class PredictionRequest(BaseModel):
    region: int
    building_type: int
    object_type: int = 0
    rooms: int
    area: float


class PredictionResponse(BaseModel):
    predicted_price: float
    price_range_low: float
    price_range_high: float
    model_type: str


@app.get("/health")
def health():
    return {"status": "ok", "version": "0.1.0"}


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    return PredictionResponse(
        predicted_price=0.0,
        price_range_low=0.0,
        price_range_high=0.0,
        model_type="placeholder",
    )
