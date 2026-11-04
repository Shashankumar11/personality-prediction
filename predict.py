"""Score one behavioral record with the trained personality model."""

import argparse
import json

import pandas as pd

import joblib

from generate_data import CATEGORICAL_FEATURES, NUMERIC_FEATURES
from train import MODEL_PATH, train


def predict_row(record: dict) -> str:
    if not MODEL_PATH.exists():
        train()
    pipeline = joblib.load(MODEL_PATH)
    frame = pd.DataFrame([record])
    return str(pipeline.predict(frame[NUMERIC_FEATURES + CATEGORICAL_FEATURES])[0])


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict a personality type.")
    parser.add_argument("--age", type=int, default=21)
    parser.add_argument("--gender", default="Female")
    parser.add_argument("--interest", default="Sports")
    parser.add_argument("--social-events", type=int, default=10)
    parser.add_argument("--hours-alone", type=float, default=2.0)
    parser.add_argument("--friends", type=int, default=16)
    parser.add_argument("--posts", type=int, default=7)
    parser.add_argument("--outdoor-days", type=int, default=5)
    parser.add_argument("--stage-fear", choices=["Yes", "No"], default="No")
    parser.add_argument("--drained", choices=["Yes", "No"], default="No")
    args = parser.parse_args()
    record = {
        "age": args.age,
        "gender": args.gender,
        "interest": args.interest,
        "social_events_per_month": args.social_events,
        "hours_alone_per_day": args.hours_alone,
        "friends_circle_size": args.friends,
        "posts_per_week": args.posts,
        "outdoor_days_per_week": args.outdoor_days,
        "stage_fear": args.stage_fear,
        "drained_after_socializing": args.drained,
    }
    print(json.dumps({"personality": predict_row(record), "input": record}, indent=2))


if __name__ == "__main__":
    main()
