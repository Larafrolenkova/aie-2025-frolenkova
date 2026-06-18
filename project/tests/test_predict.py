from pathlib import Path

import pytest

from src.predict import predict_trip_duration


def test_predict_raises_without_model(tmp_path: Path):
    payload = {
        "vendor_id": 1,
        "pickup_datetime": "2016-01-01T12:00:00",
        "passenger_count": 1,
        "pickup_longitude": -73.9855,
        "pickup_latitude": 40.7580,
        "dropoff_longitude": -73.9730,
        "dropoff_latitude": 40.7648,
        "store_and_fwd_flag": "N",
    }

    with pytest.raises(FileNotFoundError):
        predict_trip_duration(
            payload,
            model_path=tmp_path / "missing.joblib",
        )
