import os
import tempfile
import unittest

import pandas as pd

from src.data_preprocessing import build_feature_frame, interest_match, load_destinations
from src.ml_recommendation import load_model, score_destinations, train_model


class TestPreprocessing(unittest.TestCase):
    def test_interest_match(self):
        self.assertEqual(interest_match(["beaches", "hills"], "beaches|nature"), 0.5)
        self.assertEqual(interest_match(["history"], "beaches"), 0.0)
        self.assertEqual(interest_match([], "beaches"), 0.0)

    def test_cost_formula(self):
        d = load_destinations("data/destinations.csv")
        row = d.iloc[0]
        self.assertEqual(row["estimated_total_cost"], row["travel_cost"] + row["daily_cost"] * row["recommended_days"])


class TestModel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp()
        cls.model_path = os.path.join(cls.tmp, "m.joblib")
        cls.metrics = train_model("data/preference_data.csv", cls.model_path,
                                  os.path.join(cls.tmp, "m.json"))

    def test_metrics_exist_and_beat_baseline(self):
        self.assertGreater(self.metrics["accuracy"], self.metrics["majority_class_baseline_accuracy"])

    def test_scores_in_range_and_sensible(self):
        model = load_model(self.model_path)
        dest = load_destinations("data/destinations.csv")
        f = build_feature_frame(dest, 25000, ["beaches"], 4, 13.08, 80.27)
        s = score_destinations(model, f)
        self.assertTrue(((s >= 0) & (s <= 1)).all())
        # A cheap beach destination must outrank an unaffordable, non-matching one
        mahabs = s[f["destination_name"] == "Mahabalipuram"].iloc[0]
        manali = s[f["destination_name"] == "Manali"].iloc[0]
        self.assertGreater(mahabs, manali)



class TestRecommender(unittest.TestCase):
    def test_end_to_end_selection_respects_budget_and_rules(self):
        from src.recommender import recommend, validate_inputs
        tmp = tempfile.mkdtemp()
        mp = os.path.join(tmp, "m.joblib")
        train_model("data/preference_data.csv", mp, os.path.join(tmp, "m.json"))
        out = recommend(load_destinations("data/destinations.csv"), load_model(mp),
                        25000, ["beaches", "history"], 4, 13.0827, 80.2707)
        sel = out["selected"]
        self.assertLessEqual(sel["estimated_total_cost"].sum(), 25000)
        self.assertTrue((sel["recommended_days"] <= 4).all())
        self.assertTrue((sel["interest_match"] > 0).all())
        self.assertEqual(validate_inputs("", 500, [], 0).__len__(), 4)
        self.assertEqual(validate_inputs("Chennai", 5000, ["hills"], 2), [])


if __name__ == "__main__":
    unittest.main()
