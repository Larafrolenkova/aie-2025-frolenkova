from __future__ import annotations

from pathlib import Path
from threading import Lock

import joblib
import numpy as np
import pandas as pd

from src.config import load_config
from src.features import MODEL_FEATURES, add_features


_model = None
_model_lock = Lock()


def load_model(model_path: str | Path | None = None):
    """Load and cache the trained pipeline."""
    global _model
    if _model is not None:
        return _model

    with _model_lock:
        if _model is None:
            config = load_config()
            path = Path(model_path or config["paths"]["model_artifact"])
            if not path.exists():
                raise FileNotFoundError(
                    f"Model artifact not found: {path}. "
                    "Run `python -m src.train` first."
                )
            _model = joblib.load(path)
    return _model


def predict_trip_duration(
    input_data: dict,
    model_path: str | Path | None = None,
) -> dict[str, float]:
    """Generate a trip-duration prediction from raw request fields."""
    raw_frame = pd.DataFrame([input_data])
    feature_frame = add_features(raw_frame)[MODEL_FEATURES]

    pipeline = load_model(model_path)
    prediction_log = float(pipeline.predict(feature_frame)[0])
    prediction_seconds = max(float(np.expm1(prediction_log)), 0.0)

    return {
        "predicted_duration_seconds": round(prediction_seconds, 2),
        "predicted_duration_minutes": round(prediction_seconds / 60, 2),
    }
