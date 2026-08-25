import pandas as pd

from ticket_triage.batch import to_csv_bytes, triage_batch
from ticket_triage.guidance import suggest_next_action
from ticket_triage.modeling import train_models


def test_suggestion_escalates_uncertain_result():
    result = suggest_next_action("billing", "high", uncertain=True)
    assert "manual review" in result.lower()


def test_triage_batch_preserves_source_columns_and_adds_results():
    models = train_models(pd.read_csv("data/sample_tickets.csv"))
    source = pd.DataFrame({"ticket_id": [1], "message": ["The app crashes on launch"]})
    result = triage_batch(models, source)
    assert result.loc[0, "ticket_id"] == 1
    assert {
        "predicted_category",
        "category_confidence",
        "predicted_priority",
        "priority_confidence",
        "uncertain",
        "suggested_action",
    }.issubset(result.columns)
    assert b"predicted_category" in to_csv_bytes(result)

