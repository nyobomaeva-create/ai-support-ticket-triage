"""AI support ticket triage package."""

from ticket_triage.batch import to_csv_bytes, triage_batch
from ticket_triage.guidance import suggest_next_action
from ticket_triage.modeling import (
    Prediction,
    TriageModels,
    confusion_matrices,
    evaluate_models,
    predict_ticket,
    train_models,
)
from ticket_triage.validation import validate_batch_frame, validate_message

__all__ = [
    "Prediction",
    "TriageModels",
    "confusion_matrices",
    "evaluate_models",
    "predict_ticket",
    "suggest_next_action",
    "to_csv_bytes",
    "train_models",
    "triage_batch",
    "validate_batch_frame",
    "validate_message",
]
