"""Locust scenarios for the CIAN prediction API."""

from __future__ import annotations

from itertools import cycle

from locust import HttpUser, between, task

PAYLOADS = [
    {
        "rooms_count": 2,
        "total_meters": 58,
        "floor": 5,
        "floors_count": 12,
        "district": "Василеостровский",
        "underground": "Приморская",
        "author_type": "real_estate_agent",
        "street": "",
        "house_number": "",
    },
    {
        "rooms_count": 0,
        "total_meters": 32,
        "floor": 8,
        "floors_count": 16,
        "district": "Адмиралтейский",
        "underground": "Садовая",
        "author_type": "developer",
        "street": "",
        "house_number": "",
    },
    {
        "rooms_count": 3,
        "total_meters": 82,
        "floor": 4,
        "floors_count": 9,
        "district": "Петроградский",
        "underground": "Петроградская",
        "author_type": "realtor",
        "street": "",
        "house_number": "",
    },
    {
        "rooms_count": 1,
        "total_meters": 41,
        "floor": 2,
        "floors_count": 5,
        "district": "Калининский",
        "underground": "Площадь Мужества",
        "author_type": "homeowner",
        "street": "",
        "house_number": "",
    },
]


class CianPredictionUser(HttpUser):
    """User model that repeatedly calls FastAPI /predict."""

    wait_time = between(0.05, 0.20)

    def on_start(self) -> None:
        self.payloads = cycle(PAYLOADS)

    @task
    def predict_price(self) -> None:
        payload = next(self.payloads)
        with self.client.post("/predict", json=payload, catch_response=True, name="/predict") as response:
            if response.status_code != 200:
                response.failure(f"HTTP {response.status_code}: {response.text[:200]}")
                return
            try:
                body = response.json()
            except ValueError as exc:
                response.failure(f"invalid json: {exc}")
                return

            if body.get("predicted_price", 0) <= 0:
                response.failure("predicted_price is missing or non-positive")
            elif not body.get("model_name"):
                response.failure("model_name is missing")
            else:
                response.success()
