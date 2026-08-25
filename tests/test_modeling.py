import pandas as pd

from ticket_triage.modeling import confusion_matrices, evaluate_models, predict_ticket, train_models


def load_training_data():
    return pd.read_csv("data/sample_tickets.csv")


def test_train_and_predict_returns_supported_labels():
    models = train_models(load_training_data())
    result = predict_ticket(models, "I was charged twice and need an urgent refund")
    assert result.category in {"billing", "technical", "account", "general"}
    assert result.priority in {"low", "medium", "high"}
    assert 0.0 <= result.category_confidence <= 1.0
    assert 0.0 <= result.priority_confidence <= 1.0
    assert len(result.influential_terms) <= 5


def test_known_urgent_ticket_is_high_priority():
    models = train_models(load_training_data())
    result = predict_ticket(models, "Urgent: I was charged twice and need a refund immediately")
    assert result.priority == "high"


def test_evaluation_returns_required_metrics():
    frame = load_training_data()
    metrics = evaluate_models(train_models(frame), frame)
    assert set(metrics) == {
        "category_accuracy",
        "category_macro_f1",
        "priority_accuracy",
        "priority_macro_f1",
    }
    assert all(0.0 <= value <= 1.0 for value in metrics.values())


def test_confusion_matrices_use_supported_labels():
    frame = load_training_data()
    matrices = confusion_matrices(train_models(frame), frame)
    assert matrices["category"].index.tolist() == ["billing", "technical", "account", "general"]
    assert matrices["priority"].columns.tolist() == ["low", "medium", "high"]
