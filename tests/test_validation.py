from pathlib import Path

import pandas as pd
import pytest

from ticket_triage.constants import CATEGORIES, MAX_MESSAGE_LENGTH, PRIORITIES
from ticket_triage.validation import validate_batch_frame, validate_message


def test_validate_message_strips_whitespace():
    assert validate_message("  I cannot sign in  ") == "I cannot sign in"


@pytest.mark.parametrize("value", ["", "   "])
def test_validate_message_rejects_empty_values(value):
    with pytest.raises(ValueError):
        validate_message(value)


@pytest.mark.parametrize("value", [None, 42])
def test_validate_message_rejects_non_text_values(value):
    with pytest.raises(ValueError):
        validate_message(value)


def test_validate_message_rejects_excessive_length():
    with pytest.raises(ValueError, match="2,000"):
        validate_message("x" * (MAX_MESSAGE_LENGTH + 1))


def test_validate_batch_frame_requires_message_column():
    with pytest.raises(ValueError, match="message"):
        validate_batch_frame(pd.DataFrame({"text": ["hello"]}))


def test_validate_batch_frame_returns_clean_copy():
    source = pd.DataFrame({"message": ["  Refund missing ", "Cannot login"]})
    result = validate_batch_frame(source)
    assert result["message"].tolist() == ["Refund missing", "Cannot login"]
    assert source.loc[0, "message"] == "  Refund missing "


def test_sample_dataset_contract():
    path = Path("data/sample_tickets.csv")
    frame = pd.read_csv(path)
    assert list(frame.columns) == ["message", "category", "priority"]
    assert len(frame) >= 60
    assert set(frame["category"]) == set(CATEGORIES)
    assert set(frame["priority"]) == set(PRIORITIES)
    assert frame["message"].nunique() == len(frame)
    assert frame.groupby("category").size().min() >= 15


def test_streamlit_entrypoint_exists_and_has_required_tabs():
    source = Path("app.py").read_text(encoding="utf-8")
    assert "Single ticket" in source
    assert "Batch CSV" in source
    assert "st.cache_resource" in source
