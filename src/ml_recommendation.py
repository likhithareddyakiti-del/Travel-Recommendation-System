"""
src/ml_recommendation.py
------------------------
Random Forest classifier that outputs a SUITABILITY SCORE for each destination.

How the score is obtained
  * The forest has many decision trees. Each tree votes "suitable" (1) or not (0)
    for a (user request, destination) row.
  * predict_proba() returns the average vote -> a number between 0 and 1.
    That number is the suitability score (0.80 means ~80% of the trees say "suitable").
  * It is a model-confidence score, NOT a real probability that a real tourist
    would enjoy the trip.

HONESTY NOTE: the training labels in data/preference_data.csv are produced by a
hand-written rule (see generate_sample_data.py). High accuracy therefore only means
"the forest re-learned our rule". It says nothing about real-world accuracy.

Leakage control: rows are split BY USER (GroupShuffleSplit), so no user appears in
both training and test sets. The test set is never used for fitting.
"""
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import GroupShuffleSplit

from src.data_preprocessing import FEATURE_COLUMNS

MODEL_PATH = "models/suitability_model.joblib"
METRICS_PATH = "outputs/model_metrics.json"


def load_training_data(path="data/preference_data.csv"):
    df = pd.read_csv(path)
    needed = FEATURE_COLUMNS + ["suitable", "user_id"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"preference_data.csv is missing columns: {missing}")
    df = df.dropna(subset=needed).drop_duplicates()
    df = df[df["suitable"].isin([0, 1])]
    if df["suitable"].nunique() < 2:
        raise ValueError("Training data needs both suitable=1 and suitable=0 examples")
    return df.reset_index(drop=True)


def train_model(data_path="data/preference_data.csv", model_path=MODEL_PATH,
                metrics_path=METRICS_PATH, seed=42):
    """Train, evaluate on held-out users, save model + metrics. Returns metrics dict."""
    df = load_training_data(data_path)
    X, y, groups = df[FEATURE_COLUMNS], df["suitable"], df["user_id"]

    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    train_idx, test_idx = next(splitter.split(X, y, groups))
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    assert set(groups.iloc[train_idx]).isdisjoint(set(groups.iloc[test_idx])), "user leakage!"

    model = RandomForestClassifier(n_estimators=150, max_depth=8, min_samples_leaf=5,
                                   random_state=seed, n_jobs=-1)
    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    majority = int(y_train.mode()[0])
    metrics = {
        "data_note": "Labels are SYNTHETIC / rule-generated. Metrics do NOT show real-world accuracy.",
        "train_rows": int(len(X_train)), "test_rows": int(len(X_test)),
        "accuracy": round(float(accuracy_score(y_test, pred)), 4),
        "precision": round(float(precision_score(y_test, pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_test, pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y_test, pred, zero_division=0)), 4),
        "majority_class_baseline_accuracy": round(float((y_test == majority).mean()), 4),
        "confusion_matrix_[[TN,FP],[FN,TP]]": confusion_matrix(y_test, pred).tolist(),
        "feature_importance": {c: round(float(v), 4)
                               for c, v in zip(FEATURE_COLUMNS, model.feature_importances_)},
    }

    Path(model_path).parent.mkdir(parents=True, exist_ok=True)
    Path(metrics_path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "feature_columns": FEATURE_COLUMNS}, model_path)
    Path(metrics_path).write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


def load_model(model_path=MODEL_PATH):
    bundle = joblib.load(model_path)
    return bundle["model"]


def score_destinations(model, feature_df):
    """Return a Series of suitability scores (0-1), one per row of feature_df."""
    proba = model.predict_proba(feature_df[FEATURE_COLUMNS])
    positive_col = list(model.classes_).index(1)
    return pd.Series(proba[:, positive_col], index=feature_df.index, name="suitability")


if __name__ == "__main__":
    print(json.dumps(train_model(), indent=2))
