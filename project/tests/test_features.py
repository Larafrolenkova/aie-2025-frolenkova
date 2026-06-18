import pandas as pd

from src.features import add_features


def test_add_features_calculates_distance():
    frame = pd.DataFrame(
        [
            {
                "vendor_id": 1,
                "pickup_datetime": "2016-01-01 12:00:00",
                "passenger_count": 1,
                "pickup_longitude": -73.9855,
                "pickup_latitude": 40.7580,
                "dropoff_longitude": -73.9730,
                "dropoff_latitude": 40.7648,
                "store_and_fwd_flag": "N",
            }
        ]
    )

    result = add_features(frame)

    assert result.loc[0, "distance_km"] > 0
    assert result.loc[0, "pickup_hour"] == 12
    assert "dropoff_datetime" not in result.columns
