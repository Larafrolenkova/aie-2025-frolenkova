from __future__ import annotations

import numpy as np
import pandas as pd


EARTH_RADIUS_KM = 6371.0


def haversine_distance_km(
    pickup_latitude: pd.Series,
    pickup_longitude: pd.Series,
    dropoff_latitude: pd.Series,
    dropoff_longitude: pd.Series,
) -> pd.Series:
    """Calculate great-circle distance between pickup and dropoff coordinates."""
    pickup_lat = np.radians(pickup_latitude.astype(float))
    pickup_lon = np.radians(pickup_longitude.astype(float))
    dropoff_lat = np.radians(dropoff_latitude.astype(float))
    dropoff_lon = np.radians(dropoff_longitude.astype(float))

    delta_lat = dropoff_lat - pickup_lat
    delta_lon = dropoff_lon - pickup_lon

    value = (
        np.sin(delta_lat / 2) ** 2
        + np.cos(pickup_lat)
        * np.cos(dropoff_lat)
        * np.sin(delta_lon / 2) ** 2
    )
    value = np.clip(value, 0.0, 1.0)

    return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(value))


def add_features(data: pd.DataFrame) -> pd.DataFrame:
    """Create only features available before a trip is completed."""
    required = {
        "vendor_id",
        "pickup_datetime",
        "passenger_count",
        "pickup_longitude",
        "pickup_latitude",
        "dropoff_longitude",
        "dropoff_latitude",
        "store_and_fwd_flag",
    }
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    result = data.copy()
    result["pickup_datetime"] = pd.to_datetime(
        result["pickup_datetime"],
        errors="coerce",
    )

    result["pickup_hour"] = result["pickup_datetime"].dt.hour
    result["pickup_day_of_week"] = result["pickup_datetime"].dt.dayofweek
    result["pickup_month"] = result["pickup_datetime"].dt.month
    result["pickup_day"] = result["pickup_datetime"].dt.day
    result["is_weekend"] = (result["pickup_day_of_week"] >= 5).astype(int)

    result["distance_km"] = haversine_distance_km(
        result["pickup_latitude"],
        result["pickup_longitude"],
        result["dropoff_latitude"],
        result["dropoff_longitude"],
    )

    result["latitude_difference"] = (
        result["dropoff_latitude"] - result["pickup_latitude"]
    )
    result["longitude_difference"] = (
        result["dropoff_longitude"] - result["pickup_longitude"]
    )
    result["manhattan_distance_km"] = (
        np.abs(result["latitude_difference"]) * 111.0
        + np.abs(result["longitude_difference"])
        * 111.0
        * np.cos(np.radians(result["pickup_latitude"]))
    )

    return result


NUMERIC_FEATURES = [
    "passenger_count",
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
    "pickup_hour",
    "pickup_day_of_week",
    "pickup_month",
    "pickup_day",
    "is_weekend",
    "distance_km",
    "latitude_difference",
    "longitude_difference",
    "manhattan_distance_km",
]

CATEGORICAL_FEATURES = [
    "vendor_id",
    "store_and_fwd_flag",
]

MODEL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
