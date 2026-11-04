"""Create a reproducible behavioral dataset for personality classification."""

from pathlib import Path

import numpy as np
import pandas as pd

COLUMNS = [
    "age",
    "gender",
    "interest",
    "social_events_per_month",
    "hours_alone_per_day",
    "friends_circle_size",
    "posts_per_week",
    "outdoor_days_per_week",
    "stage_fear",
    "drained_after_socializing",
    "personality",
]

NUMERIC_FEATURES = [
    "age",
    "social_events_per_month",
    "hours_alone_per_day",
    "friends_circle_size",
    "posts_per_week",
    "outdoor_days_per_week",
]
CATEGORICAL_FEATURES = [
    "gender",
    "interest",
    "stage_fear",
    "drained_after_socializing",
]
TARGET = "personality"
LABELS = ("Introvert", "Ambivert", "Extrovert")


def generate(n_rows: int = 1500, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    gender = rng.choice(["Female", "Male", "Other"], n_rows)
    interest = rng.choice(
        ["Sports", "Books", "Art", "Technology", "Music"], n_rows
    )
    stage_fear = rng.choice(["Yes", "No"], n_rows)
    drained = rng.choice(["Yes", "No"], n_rows)
    social = rng.integers(0, 15, n_rows)
    hours_alone = rng.uniform(0, 12, n_rows).round(2)
    friends = rng.integers(0, 30, n_rows)
    posts = rng.integers(0, 20, n_rows)
    outdoor = rng.integers(0, 8, n_rows)
    age = rng.integers(18, 30, n_rows)

    score = (
        0.22 * social
        + 0.08 * friends
        + 0.12 * posts
        + 0.28 * outdoor
        - 0.35 * hours_alone
        + np.where(stage_fear == "Yes", -1.4, 0.6)
        + np.where(drained == "Yes", -1.1, 0.5)
        + np.select(
            [interest == "Sports", interest == "Books", interest == "Art"],
            [0.8, -0.7, -0.3],
            default=0.1,
        )
        + rng.normal(0, 0.45, n_rows)
    )
    order = np.argsort(score)
    personality = np.empty(n_rows, dtype=object)
    third = n_rows // 3
    personality[order[:third]] = "Introvert"
    personality[order[third : 2 * third]] = "Ambivert"
    personality[order[2 * third :]] = "Extrovert"
    flips = rng.random(n_rows) < 0.04
    swapped = rng.choice(list(LABELS), n_rows)
    personality = np.where(flips, swapped, personality)
    return pd.DataFrame(
        {
            "age": age,
            "gender": gender,
            "interest": interest,
            "social_events_per_month": social,
            "hours_alone_per_day": hours_alone,
            "friends_circle_size": friends,
            "posts_per_week": posts,
            "outdoor_days_per_week": outdoor,
            "stage_fear": stage_fear,
            "drained_after_socializing": drained,
            "personality": personality,
        }
    )


def write_dataset(path: Path, n_rows: int = 1500, seed: int = 42) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    generate(n_rows, seed).to_csv(path, index=False)
    return path


if __name__ == "__main__":
    destination = Path(__file__).resolve().parent / "data" / "personality.csv"
    write_dataset(destination)
    print(f"Wrote {destination}")
