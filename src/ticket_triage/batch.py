"""Batch support-ticket processing and portable CSV output."""

import pandas as pd

from ticket_triage.guidance import suggest_next_action
from ticket_triage.modeling import TriageModels, predict_ticket
from ticket_triage.validation import validate_batch_frame


def triage_batch(models: TriageModels, frame: pd.DataFrame) -> pd.DataFrame:
    """Preserve source columns and append predictions for every ticket."""
    result = validate_batch_frame(frame)
    predictions = [predict_ticket(models, message) for message in result["message"]]
    result["predicted_category"] = [prediction.category for prediction in predictions]
    result["category_confidence"] = [
        round(prediction.category_confidence, 4) for prediction in predictions
    ]
    result["predicted_priority"] = [prediction.priority for prediction in predictions]
    result["priority_confidence"] = [
        round(prediction.priority_confidence, 4) for prediction in predictions
    ]
    result["uncertain"] = [prediction.uncertain for prediction in predictions]
    result["suggested_action"] = [
        suggest_next_action(
            prediction.category,
            prediction.priority,
            prediction.uncertain,
        )
        for prediction in predictions
    ]
    return result


def to_csv_bytes(frame: pd.DataFrame) -> bytes:
    """Serialize a result frame for a browser download."""
    return frame.to_csv(index=False).encode("utf-8")

