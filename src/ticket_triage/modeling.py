"""Training, prediction, evaluation, and lightweight explanations."""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.pipeline import Pipeline

from ticket_triage.constants import CATEGORIES, LOW_CONFIDENCE_THRESHOLD, PRIORITIES
from ticket_triage.validation import validate_message


@dataclass(frozen=True)
class TriageModels:
    """The independently trained category and priority pipelines."""

    category_model: Pipeline
    priority_model: Pipeline


@dataclass(frozen=True)
class Prediction:
    """A serializable ticket prediction for UI and batch consumers."""

    category: str
    category_confidence: float
    priority: str
    priority_confidence: float
    uncertain: bool
    influential_terms: tuple[str, ...]


def _build_pipeline() -> Pipeline:
    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True),
            ),
            (
                "classifier",
                LogisticRegression(
                    max_iter=1_000,
                    class_weight="balanced",
                    random_state=42,
                ),
            ),
        ]
    )


def train_models(frame: pd.DataFrame) -> TriageModels:
    """Train deterministic category and priority classifiers."""
    required = {"message", "category", "priority"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Training data is missing columns: {', '.join(sorted(missing))}.")

    category_model = _build_pipeline().fit(frame["message"], frame["category"])
    priority_model = _build_pipeline().fit(frame["message"], frame["priority"])
    return TriageModels(category_model=category_model, priority_model=priority_model)


def _predict_label_and_confidence(model: Pipeline, message: str) -> tuple[str, float]:
    probabilities = model.predict_proba([message])[0]
    index = int(np.argmax(probabilities))
    classifier = model.named_steps["classifier"]
    return str(classifier.classes_[index]), float(probabilities[index])


def _influential_terms(model: Pipeline, message: str, label: str) -> tuple[str, ...]:
    vectorizer = model.named_steps["tfidf"]
    classifier = model.named_steps["classifier"]
    vector = vectorizer.transform([message])
    label_index = int(np.where(classifier.classes_ == label)[0][0])
    contributions = vector.toarray()[0] * classifier.coef_[label_index]
    present_indices = np.flatnonzero(vector.toarray()[0])
    ranked = sorted(present_indices, key=lambda index: contributions[index], reverse=True)
    positive = [index for index in ranked if contributions[index] > 0]
    selected = positive[:5] if positive else ranked[:5]
    features = vectorizer.get_feature_names_out()
    return tuple(str(features[index]) for index in selected)


def predict_ticket(models: TriageModels, message: str) -> Prediction:
    """Classify one validated support message."""
    cleaned = validate_message(message)
    category, category_confidence = _predict_label_and_confidence(
        models.category_model, cleaned
    )
    priority, priority_confidence = _predict_label_and_confidence(
        models.priority_model, cleaned
    )
    uncertain = min(category_confidence, priority_confidence) < LOW_CONFIDENCE_THRESHOLD
    return Prediction(
        category=category,
        category_confidence=category_confidence,
        priority=priority,
        priority_confidence=priority_confidence,
        uncertain=uncertain,
        influential_terms=_influential_terms(models.category_model, cleaned, category),
    )


def evaluate_models(models: TriageModels, frame: pd.DataFrame) -> dict[str, float]:
    """Return training-set smoke metrics; these are not unbiased production estimates."""
    category_predictions = models.category_model.predict(frame["message"])
    priority_predictions = models.priority_model.predict(frame["message"])
    return {
        "category_accuracy": float(accuracy_score(frame["category"], category_predictions)),
        "category_macro_f1": float(
            f1_score(frame["category"], category_predictions, average="macro")
        ),
        "priority_accuracy": float(accuracy_score(frame["priority"], priority_predictions)),
        "priority_macro_f1": float(
            f1_score(frame["priority"], priority_predictions, average="macro")
        ),
    }


def confusion_matrices(models: TriageModels, frame: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Return labeled training-set confusion matrices for transparent inspection."""
    category_predictions = models.category_model.predict(frame["message"])
    priority_predictions = models.priority_model.predict(frame["message"])
    return {
        "category": pd.DataFrame(
            confusion_matrix(frame["category"], category_predictions, labels=CATEGORIES),
            index=CATEGORIES,
            columns=CATEGORIES,
        ),
        "priority": pd.DataFrame(
            confusion_matrix(frame["priority"], priority_predictions, labels=PRIORITIES),
            index=PRIORITIES,
            columns=PRIORITIES,
        ),
    }

