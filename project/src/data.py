from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.features import add_features


def load_csv(path: str | Path, max_rows: int | None = None) -> pd.DataFrame:
    """Load a CSV file with an explicit existence check."""
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(
            f"Data file not found: {file_path}. "
            "See data/README.md for download instructions."
        )
    return pd.read_csv(file_path, nrows=max_rows)


def prepare_training_data(
    data: pd.DataFrame,
    filters: dict,
    target_column: str = "trip_duration",
) -> pd.DataFrame:
    """Create features and remove clearly invalid training rows."""
    if target_column not in data.columns:
        raise ValueError(f"Target column not found: {target_column}")

    result = add_features(data)
    result = result.dropna(
        subset=[
            target_column,
            "pickup_datetime",
            "distance_km",
            "passenger_count",
        ]
    )

    result = result[
        result[target_column].between(
            filters["min_trip_duration_seconds"],
            filters["max_trip_duration_seconds"],
        )
        & result["passenger_count"].between(
            filters["min_passenger_count"],
            filters["max_passenger_count"],
        )
        & result["distance_km"].between(
            filters["min_distance_km"],
            filters["max_distance_km"],
        )
    ]

    return result.reset_index(drop=True)
