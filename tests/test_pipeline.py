import json
import unittest
from pathlib import Path

from generate_data import LABELS, generate
from predict import predict_row
from train import train

ROOT = Path(__file__).resolve().parents[1]


class PersonalityPipelineTest(unittest.TestCase):
    def test_training_encodes_scales_and_selects_features(self):
        csv_path = ROOT / "data" / "personality.csv"
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        generate().to_csv(csv_path, index=False)

        metrics = train(csv_path)

        self.assertGreaterEqual(metrics["accuracy"], 0.75)
        self.assertEqual(metrics["model"], "logistic_regression")
        self.assertGreaterEqual(len(metrics["features_selected"]), 1)
        self.assertGreater(metrics["anova"][0]["f_score"], metrics["anova"][-1]["f_score"])
        saved = json.loads((ROOT / "models" / "metrics.json").read_text(encoding="utf-8"))
        self.assertAlmostEqual(saved["accuracy"], metrics["accuracy"])

    def test_prediction_returns_a_known_label(self):
        label = predict_row(
            {
                "age": 21,
                "gender": "Male",
                "interest": "Books",
                "social_events_per_month": 1,
                "hours_alone_per_day": 10,
                "friends_circle_size": 2,
                "posts_per_week": 0,
                "outdoor_days_per_week": 0,
                "stage_fear": "Yes",
                "drained_after_socializing": "Yes",
            }
        )
        self.assertIn(label, LABELS)
        outgoing = predict_row(
            {
                "age": 22,
                "gender": "Female",
                "interest": "Sports",
                "social_events_per_month": 12,
                "hours_alone_per_day": 1,
                "friends_circle_size": 20,
                "posts_per_week": 10,
                "outdoor_days_per_week": 6,
                "stage_fear": "No",
                "drained_after_socializing": "No",
            }
        )
        self.assertEqual(label, "Introvert")
        self.assertEqual(outgoing, "Extrovert")


if __name__ == "__main__":
    unittest.main()
