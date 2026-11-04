"""Train a logistic regression personality classifier."""

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer

from generate_data import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    TARGET,
    write_dataset,
)

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "personality.csv"
MODEL_PATH = ROOT / "models" / "personality_model.joblib"
METRICS_PATH = ROOT / "models" / "metrics.json"


def load_frame(csv_path: Path = DATA_PATH) -> pd.DataFrame:
    if not csv_path.exists():
        write_dataset(csv_path)
    frame = pd.read_csv(csv_path)
    missing = [column for column in NUMERIC_FEATURES + CATEGORICAL_FEATURES + [TARGET] if column not in frame.columns]
    if missing:
        raise ValueError(f"Dataset is missing columns: {', '.join(missing)}")
    return frame


def build_pipeline(k_features: int = 8) -> Pipeline:
    preprocess = ColumnTransformer(
        transformers=[
            ("numeric", StandardScaler(), NUMERIC_FEATURES),
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore"),
                CATEGORICAL_FEATURES,
            ),
        ]
    )
    return Pipeline(
        steps=[
            ("preprocess", preprocess),
            ("anova", SelectKBest(score_func=f_classif, k=k_features)),
            ("model", LogisticRegression(max_iter=1000, class_weight="balanced")),
        ]
    )


def train(csv_path: Path = DATA_PATH, random_state: int = 42) -> dict:
    frame = load_frame(csv_path)
    features = frame[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    labels = frame[TARGET]
    x_train, x_test, y_train, y_test = train_test_split(
        features,
        labels,
        test_size=0.2,
        random_state=random_state,
        stratify=labels,
    )
    pipeline = build_pipeline()
    pipeline.fit(x_train, y_train)
    predictions = pipeline.predict(x_test)
    accuracy = float(accuracy_score(y_test, predictions))

    feature_names = pipeline.named_steps["preprocess"].get_feature_names_out()
    selector = pipeline.named_steps["anova"]
    selected = feature_names[selector.get_support()].tolist()
    anova = [
        {"feature": name, "f_score": float(score)}
        for name, score in sorted(
            zip(feature_names, selector.scores_),
            key=lambda item: item[1],
            reverse=True,
        )
    ]
    metrics = {
        "accuracy": accuracy,
        "model": "logistic_regression",
        "test_rows": int(len(y_test)),
        "features_selected": selected,
        "anova": anova,
        "classification_report": classification_report(y_test, predictions),
    }
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)
    METRICS_PATH.write_text(json.dumps({k: v for k, v in metrics.items() if k != "classification_report"}, indent=2), encoding="utf-8")
    return metrics


if __name__ == "__main__":
    result = train()
    print(f"Test accuracy: {result['accuracy']:.4f}")
    print("ANOVA feature scores (highest first):")
    for row in result["anova"][:8]:
        print(f"  {row['feature']}: {row['f_score']:.2f}")
    print(result["classification_report"])
