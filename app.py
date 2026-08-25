"""Streamlit interface for AI Support Ticket Triage."""

from pathlib import Path

import pandas as pd
import streamlit as st

from ticket_triage import (
    confusion_matrices,
    evaluate_models,
    predict_ticket,
    suggest_next_action,
    to_csv_bytes,
    train_models,
    triage_batch,
)

ROOT = Path(__file__).parent
DATA_PATH = ROOT / "data" / "sample_tickets.csv"
BATCH_EXAMPLE_PATH = ROOT / "data" / "batch_example.csv"

EXAMPLES = {
    "Billing issue": "I was charged twice and need an urgent refund.",
    "Technical failure": "The app crashes whenever I upload a PDF file.",
    "Account access": "I cannot access my account after changing my phone number.",
    "General question": "Where can I find documentation for team permissions?",
}


@st.cache_resource
def load_resources():
    """Load demonstration data and train deterministic local models once."""
    frame = pd.read_csv(DATA_PATH)
    return frame, train_models(frame)


def format_label(label: str) -> str:
    """Convert a machine label into display text."""
    return label.replace("_", " ").title()


st.set_page_config(
    page_title="AI Support Ticket Triage",
    page_icon="🎫",
    layout="wide",
)

st.markdown(
    """
    <style>
    .stApp { background: #f7f8fc; }
    [data-testid="stMetric"] {
        background: white;
        border: 1px solid #e5e7eb;
        border-radius: 14px;
        padding: 16px;
        box-shadow: 0 8px 24px rgba(15, 23, 42, 0.05);
    }
    .eyebrow {
        color: #6d5dfc;
        font-size: 0.78rem;
        font-weight: 700;
        letter-spacing: 0.12em;
        text-transform: uppercase;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

training_frame, models = load_resources()

st.markdown('<p class="eyebrow">Portfolio ML Application</p>', unsafe_allow_html=True)
st.title("AI Support Ticket Triage")
st.write(
    "Classify incoming support messages by category and priority with transparent "
    "confidence indicators and rule-based routing guidance."
)

single_tab, batch_tab = st.tabs(["Single ticket", "Batch CSV"])

with single_tab:
    st.subheader("Analyze one support message")
    example_name = st.selectbox("Start with an example", ["Custom message", *EXAMPLES])
    default_message = "" if example_name == "Custom message" else EXAMPLES[example_name]
    message = st.text_area(
        "Customer message",
        value=default_message,
        height=150,
        max_chars=2_001,
        placeholder="Describe the customer's issue...",
    )

    if st.button("Analyze ticket", type="primary", width="stretch"):
        try:
            prediction = predict_ticket(models, message)
            action = suggest_next_action(
                prediction.category,
                prediction.priority,
                prediction.uncertain,
            )
            metric_columns = st.columns(4)
            metric_columns[0].metric("Category", format_label(prediction.category))
            metric_columns[1].metric(
                "Category confidence", f"{prediction.category_confidence:.0%}"
            )
            metric_columns[2].metric("Priority", format_label(prediction.priority))
            metric_columns[3].metric(
                "Priority confidence", f"{prediction.priority_confidence:.0%}"
            )

            if prediction.uncertain:
                st.warning("Low-confidence result — manual review is recommended.")
            else:
                st.success("The model produced a result above the confidence threshold.")

            st.markdown("**Influential terms**")
            st.write(
                " · ".join(prediction.influential_terms)
                if prediction.influential_terms
                else "No influential terms were available."
            )
            st.markdown("**Suggested next action**")
            st.info(action)
        except (TypeError, ValueError) as error:
            st.error(str(error))

with batch_tab:
    st.subheader("Analyze a CSV batch")
    st.write("Upload a UTF-8 CSV containing a column named `message`.")
    st.download_button(
        "Download example CSV",
        data=BATCH_EXAMPLE_PATH.read_bytes(),
        file_name="batch_example.csv",
        mime="text/csv",
    )
    uploaded_file = st.file_uploader("Upload ticket CSV", type=["csv"])

    if uploaded_file is not None:
        try:
            uploaded_frame = pd.read_csv(uploaded_file)
            st.markdown("**Input preview**")
            st.dataframe(uploaded_frame.head(10), width="stretch")
            if st.button("Analyze CSV", type="primary", width="stretch"):
                result_frame = triage_batch(models, uploaded_frame)
                st.markdown("**Triage results**")
                st.dataframe(result_frame, width="stretch")
                st.download_button(
                    "Download analyzed results",
                    data=to_csv_bytes(result_frame),
                    file_name="triaged_tickets.csv",
                    mime="text/csv",
                    width="stretch",
                )
        except pd.errors.EmptyDataError:
            st.error("The uploaded CSV is empty.")
        except pd.errors.ParserError:
            st.error("The uploaded file is not a valid CSV.")
        except UnicodeDecodeError:
            st.error("The uploaded CSV must use UTF-8 encoding.")
        except (TypeError, ValueError) as error:
            st.error(str(error))

with st.expander("Model details and limitations"):
    metrics = evaluate_models(models, training_frame)
    metric_columns = st.columns(4)
    metric_columns[0].metric("Category accuracy", f"{metrics['category_accuracy']:.0%}")
    metric_columns[1].metric("Category macro F1", f"{metrics['category_macro_f1']:.0%}")
    metric_columns[2].metric("Priority accuracy", f"{metrics['priority_accuracy']:.0%}")
    metric_columns[3].metric("Priority macro F1", f"{metrics['priority_macro_f1']:.0%}")

    matrices = confusion_matrices(models, training_frame)
    category_column, priority_column = st.columns(2)
    category_column.markdown("**Category confusion matrix**")
    category_column.dataframe(matrices["category"], width="stretch")
    priority_column.markdown("**Priority confusion matrix**")
    priority_column.dataframe(matrices["priority"], width="stretch")

    st.caption(
        "Metrics are training-set smoke checks on a small original demonstration dataset. "
        "They are not an unbiased estimate of real-world accuracy. Confidence is not a "
        "guarantee; real support decisions require human review. Suggested actions are "
        "deterministic rules rather than LLM-generated advice."
    )
