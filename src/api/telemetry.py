"""Optional production telemetry: Prometheus metrics and Kafka events."""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from functools import lru_cache
from typing import Any

logger = logging.getLogger(__name__)

try:
    from kafka import KafkaProducer
except ModuleNotFoundError:  # pragma: no cover - optional production dependency
    KafkaProducer = None

try:
    from prometheus_client import Counter, Histogram
except ModuleNotFoundError:  # pragma: no cover - optional production dependency
    Counter = Histogram = None

KAFKA_BOOTSTRAP_ENV = "KAFKA_BOOTSTRAP_SERVERS"
KAFKA_TOPIC_ENV = "KAFKA_PREDICTION_TOPIC"
DEFAULT_TOPIC = "cian.predictions"

if Counter is not None and Histogram is not None:
    PREDICTION_COUNTER = Counter(
        "cian_prediction_requests_total",
        "Total prediction requests served by model and status.",
        ["model_name", "status"],
    )
    PREDICTION_PRICE_HISTOGRAM = Histogram(
        "cian_predicted_price_rub",
        "Predicted apartment price in RUB.",
        buckets=(5_000_000, 10_000_000, 20_000_000, 50_000_000, 100_000_000, 200_000_000, float("inf")),
    )
    PREDICTION_LATENCY = Histogram(
        "cian_prediction_latency_seconds",
        "Prediction endpoint latency in seconds.",
        buckets=(0.05, 0.1, 0.25, 0.5, 1.0, 2.5, float("inf")),
    )
else:
    PREDICTION_COUNTER = PREDICTION_PRICE_HISTOGRAM = PREDICTION_LATENCY = None


@lru_cache(maxsize=1)
def kafka_producer() -> Any | None:
    """Create a cached Kafka producer only when explicitly configured."""
    bootstrap_servers = os.getenv(KAFKA_BOOTSTRAP_ENV)
    if not bootstrap_servers or KafkaProducer is None:
        return None
    try:
        return KafkaProducer(
            bootstrap_servers=bootstrap_servers,
            value_serializer=lambda value: json.dumps(value, ensure_ascii=False).encode("utf-8"),
            linger_ms=20,
            retries=1,
        )
    except Exception:
        return None


def publish_prediction_event(payload: dict[str, Any]) -> bool:
    """Publish a prediction event to Kafka when available.

    The send is intentionally non-blocking: prediction latency should not wait
    for Kafka acknowledgements. Kafka UI/consumers may observe events a moment
    later, while the API can keep serving low-latency predictions.
    """
    producer = kafka_producer()
    if producer is None:
        return False

    topic = os.getenv(KAFKA_TOPIC_ENV, DEFAULT_TOPIC)
    event = {
        "event_type": "prediction_served",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        **payload,
    }
    try:
        producer.send(topic, event)
        logger.debug("kafka event published: topic=%s model=%s price=%.0f", topic, payload.get("model_name"), payload.get("predicted_price", 0))
        return True
    except Exception:
        logger.warning("failed to publish kafka event to topic=%s", topic, exc_info=True)
        return False


def record_prediction_metrics(
    model_name: str, predicted_price: float, latency_ms: float | None = None, status: str = "ok"
) -> None:
    """Update Prometheus metrics when prometheus-client is installed."""
    if PREDICTION_COUNTER is None:
        return
    PREDICTION_COUNTER.labels(model_name=model_name, status=status).inc()
    if status == "ok":
        PREDICTION_PRICE_HISTOGRAM.observe(predicted_price)
    if latency_ms is not None and PREDICTION_LATENCY is not None:
        PREDICTION_LATENCY.observe(latency_ms / 1000.0)
    logger.info(
        "metrics recorded: model=%s status=%s price=%.0f latency_ms=%.1f",
        model_name, status, predicted_price, latency_ms if latency_ms else -1,
    )
