from __future__ import annotations

import logging
import time

from fastapi import FastAPI, HTTPException

from src.config import load_config
from src.predict import load_model, predict_trip_duration
from src.schemas import HealthResponse, PredictionRequest, PredictionResponse


config = load_config()
logging.basicConfig(
    level=getattr(logging, config["service"]["log_level"].upper(), logging.INFO),
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="NYC Taxi Trip Duration API",
    version=config["service"]["model_version"],
)


@app.on_event("startup")
def startup_event() -> None:
    try:
        load_model()
        logger.info("Model loaded successfully")
    except FileNotFoundError as error:
        logger.warning("%s", error)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    try:
        load_model()
        loaded = True
        status = "ok"
    except FileNotFoundError:
        loaded = False
        status = "degraded"

    return HealthResponse(
        status=status,
        model_loaded=loaded,
        model_version=config["service"]["model_version"],
    )


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest) -> PredictionResponse:
    started = time.perf_counter()
    try:
        payload = request.model_dump(mode="json")
        result = predict_trip_duration(payload)
    except FileNotFoundError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
    except (ValueError, TypeError) as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    except Exception as error:
        logger.exception("Unexpected prediction error")
        raise HTTPException(
            status_code=500,
            detail="Internal prediction error",
        ) from error

    latency_ms = (time.perf_counter() - started) * 1000
    logger.info("POST /predict status=200 latency_ms=%.2f", latency_ms)

    return PredictionResponse(
        **result,
        model_version=config["service"]["model_version"],
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "src.service:app",
        host=config["service"]["host"],
        port=int(config["service"]["port"]),
        reload=False,
    )
