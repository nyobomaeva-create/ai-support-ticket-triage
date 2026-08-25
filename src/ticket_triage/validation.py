"""Input validation for individual and batch support tickets."""

import pandas as pd

from ticket_triage.constants import MAX_MESSAGE_LENGTH, MESSAGE_COLUMN


def validate_message(message: object) -> str:
    """Return a normalized message or raise a user-facing validation error."""
    if not isinstance(message, str):
        raise ValueError("Message must be text.")  # noqa: TRY004 - one public validation error
    cleaned = message.strip()
    if not cleaned:
        raise ValueError("Message cannot be empty.")
    if len(cleaned) > MAX_MESSAGE_LENGTH:
        raise ValueError("Message cannot exceed 2,000 characters.")
    return cleaned


def validate_batch_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Return a cleaned copy of a ticket frame with a required message column."""
    if MESSAGE_COLUMN not in frame.columns:
        raise ValueError("CSV must contain a 'message' column.")
    if frame.empty:
        raise ValueError("CSV must contain at least one ticket.")
    cleaned = frame.copy()
    cleaned[MESSAGE_COLUMN] = cleaned[MESSAGE_COLUMN].map(validate_message)
    return cleaned
