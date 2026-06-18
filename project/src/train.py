from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_log_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeRegressor

from src.config import load_config
from src.data import load_csv, prepare_training_data
from src.features import CATEGORICAL_FEATURES, MODEL_FEATURES, NUMERIC_FEATURES


def build_preprocessor() -> ColumnTransformer:
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OneHotEncoder(handle_unknown="ignore", drop="first"),
            ),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_transformer, NUMERIC_FEATURES),
            ("categorical", categorical_transformer, CATEGORICAL_FEATURES),
        ]
    )


def build_models(random_state: int) -> dict[str, object]:
    return {
        "dummy_median": DummyRegressor(strategy="median"),
        "linear_regression": LinearRegression(),
        "decision_tree": DecisionTreeRegressor(
            max_depth=18,
            min_samples_leaf=10,
            random_state=random_state,
        ),
        "random_forest": RandomForestRegressor(
            n_estimators=160,
            max_depth=24,
            min_samples_leaf=5,
            n_jobs=-1,
            random_state=random_state,
        ),
        "gradient_boosting": GradientBoostingRegressor(
            n_estimators=180,
            learning_rate=0.06,
            max_depth=4,
            random_state=random_state,
        ),
    }


def evaluate(
    pipeline: Pipeline,
    X_test: pd.DataFrame,
    y_test_log: pd.Series,
) -> dict[str, float]:
    start = time.perf_counter()
    prediction_log = pipeline.predict(X_test)
    inference_seconds = time.perf_counter() - start

    y_true_seconds = np.expm1(y_test_log.to_numpy())
    y_pred_seconds = np.maximum(np.expm1(prediction_log), 0)

    return {
        "rmsle": float(
            np.sqrt(
                mean_squared_log_error(
                    y_true_seconds,
                    y_pred_seconds,
                )
            )
        ),
        "mae_seconds": float(
            mean_absolute_error(
                y_true_seconds,
                y_pred_seconds,
            )
        ),
        "mae_minutes": float(
            mean_absolute_error(
                y_true_seconds,
                y_pred_seconds,
            )
            / 60
        ),
        "r2": float(r2_score(y_true_seconds, y_pred_seconds)),
        "inference_ms_per_row": float(
            inference_seconds / max(len(X_test), 1) * 1000
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("--max-rows", type=int, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    random_state = int(config["project"]["random_state"])
    max_rows = args.max_rows
    if max_rows is None:
        max_rows = config["training"].get("max_rows")

    raw_data = load_csv(config["paths"]["train_data"], max_rows=max_rows)
    prepared = prepare_training_data(
        raw_data,
        filters=config["filters"],
        target_column=config["training"]["target_column"],
    )

    X = prepared[MODEL_FEATURES]
    y_log = np.log1p(prepared[config["training"]["target_column"]])

    X_train, X_test, y_train_log, y_test_log = train_test_split(
        X,
        y_log,
        test_size=float(config["training"]["test_size"]),
        random_state=random_state,
    )

    results: list[dict] = []
    fitted_pipelines: dict[str, Pipeline] = {}

    for model_name, estimator in build_models(random_state).items():
        pipeline = Pipeline(
            steps=[
                ("preprocessor", build_preprocessor()),
                ("model", estimator),
            ]
        )

        started = time.perf_counter()
        pipeline.fit(X_train, y_train_log)
        training_seconds = time.perf_counter() - started

        metrics = evaluate(pipeline, X_test, y_test_log)
        row = {
            "model": model_name,
            "training_seconds": training_seconds,
            **metrics,
        }
        results.append(row)
        fitted_pipelines[model_name] = pipeline
        print(row)

    results_frame = pd.DataFrame(results).sort_values("rmsle")
    selected_name = str(results_frame.iloc[0]["model"])

    selected_pipeline = fitted_pipelines[selected_name]

    model_path = Path(config["paths"]["model_artifact"])
    metrics_path = Path(config["paths"]["metrics_artifact"])
    experiments_path = Path(config["paths"]["experiments_artifact"])

    for path in [model_path, metrics_path, experiments_path]:
        path.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(selected_pipeline, model_path)
    results_frame.to_csv(
        experiments_path,
        index=False,
        encoding="utf-8-sig",
    )

    selected_metrics = next(
        row for row in results if row["model"] == selected_name
    )
    metrics_payload = {
        "selected_model": selected_name,
        "model_version": config["service"]["model_version"],
        "training_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "metrics": selected_metrics,
    }
    metrics_path.write_text(
        json.dumps(metrics_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"Selected model: {selected_name}")
    print(f"Model saved to: {model_path}")
    print(f"Metrics saved to: {metrics_path}")
    print(f"Experiments saved to: {experiments_path}")


if __name__ == "__main__":
    main()
