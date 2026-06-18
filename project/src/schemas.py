from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    vendor_id: int = Field(..., ge=1, le=2)
    pickup_datetime: datetime
    passenger_count: int = Field(..., ge=1, le=8)
    pickup_longitude: float = Field(..., ge=-180, le=180)
    pickup_latitude: float = Field(..., ge=-90, le=90)
    dropoff_longitude: float = Field(..., ge=-180, le=180)
    dropoff_latitude: float = Field(..., ge=-90, le=90)
    store_and_fwd_flag: Literal["N", "Y"] = "N"


class PredictionResponse(BaseModel):
    predicted_duration_seconds: float
    predicted_duration_minutes: float
    model_version: str


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_version: str
