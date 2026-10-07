"""Train an XGBoost regressor for continuous accident risk (0-1)."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBRegressor

from config import (
    CATEGORICAL_FEATURES,
    DATA_PATH,
    METRICS_PATH,
    MODEL_DIR,
    MODEL_PATH,
    NUMERIC_FEATURES,
    RANDOM_STATE,
    TARGET_COLUMN,
    TEST_SIZE,
)
from feature_engineering import standardize_dataframe
from generate_dataset import generate_dataset, main as write_dataset


def load_accident_frame(path: Path = DATA_PATH) -> pd.DataFrame:
    if not path.is_file():
        print(f"{path.name} was not found. Generating a production-scale training corpus.")
        write_dataset()
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError(f"{path} is empty.")
    engineered = standardize_dataframe(df)
    if engineered[TARGET_COLUMN].isna().all():
        raise ValueError(
            "Could not derive a continuous risk target. Provide risk_score or a severity column."
        )
    engineered = engineered.dropna(subset=[TARGET_COLUMN]).reset_index(drop=True)
    return engineered


def build_pipeline() -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                CATEGORICAL_FEATURES,
            ),
            ("numeric", StandardScaler(), NUMERIC_FEATURES),
        ],
        remainder="drop",
    )
    regressor = XGBRegressor(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.85,
        colsample_bytree=0.85,
        min_child_weight=4,
        reg_lambda=1.2,
        objective="reg:squarederror",
        n_jobs=-1,
        random_state=RANDOM_STATE,
        tree_method="hist",
    )
    return Pipeline(steps=[("preprocessor", preprocessor), ("regressor", regressor)])


def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    return {
        "rmse": round(rmse, 5),
        "mae": round(float(mean_absolute_error(y_true, y_pred)), 5),
        "r2": round(float(r2_score(y_true, y_pred)), 5),
    }


def train_and_persist() -> dict[str, float]:
    frame = load_accident_frame()
    features = frame[CATEGORICAL_FEATURES + NUMERIC_FEATURES]
    target = frame[TARGET_COLUMN].to_numpy(dtype=float)

    x_train, x_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )

    pipeline = build_pipeline()
    pipeline.fit(x_train, y_train)

    train_pred = np.clip(pipeline.predict(x_train), 0, 1)
    test_pred = np.clip(pipeline.predict(x_test), 0, 1)
    metrics = {
        "n_rows": int(len(frame)),
        "n_features_raw": int(features.shape[1]),
        "train": evaluate(y_train, train_pred),
        "test": evaluate(y_test, test_pred),
    }

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    artifact = {
        "pipeline": pipeline,
        "categorical_features": CATEGORICAL_FEATURES,
        "numeric_features": NUMERIC_FEATURES,
        "target": TARGET_COLUMN,
        "metrics": metrics,
    }
    joblib.dump(artifact, MODEL_PATH)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print("Training complete")
    print(json.dumps(metrics, indent=2))
    print(f"Saved model bundle to {MODEL_PATH}")
    return metrics


def main() -> None:
    if not DATA_PATH.is_file():
        generate_dataset().to_csv(DATA_PATH, index=False)
    train_and_persist()


if __name__ == "__main__":
    main()
