"""
app.py
------
Streamlit web app for the Memory Lifecycle Classifier.

User flow:
1. User types or pastes a personal note.
2. App calls the trained sklearn classifier.
3. App displays predicted label, confidence chart, and an optional LLM-generated
   explanation.

To run locally:
    streamlit run app.py

Built per Lecture 5 (Vibe Coding): plan, prompt, publish, polish.
"""

import os
import sys
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "src"))

from llm_explain import explain  # noqa: E402

MODEL_PATH = ROOT / "models" / "classifier.pkl"


st.set_page_config(
    page_title="Productivity Note Classifier",
    page_icon=":card_file_box:",
    layout="centered",
)


@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        return None
    return joblib.load(MODEL_PATH)


model = load_model()


st.title("Productivity Note Classifier")
st.markdown(
    "Triage knowledge worker notes by lifecycle. Paste any work-related note and the model will "
    "tell you whether it's *ephemeral*, *actionable*, or *archive-worthy* "
    "and explain why."
)
st.markdown("---")


if model is None:
    st.error(
        "Model not found at `models/classifier.pkl`. "
        "Run `python src/train.py` first."
    )
    st.stop()

note_text = st.text_area(
    "Your note",
    height=140,
    placeholder="Example: Fix the login bug in Modun before Friday demo.",
)

use_llm = st.checkbox(
    "Use LLM to explain the prediction (optional)",
    value=False,
    help="Requires OPENAI_API_KEY in the environment.",
)

predict_clicked = st.button("Classify", type="primary", use_container_width=True)


LABEL_DESCRIPTIONS = {
    "ephemeral": "Short-lived reminder. Safe to discard within hours or days.",
    "actionable": "Concrete task. Has a clear next step, often a deadline.",
    "archive": "Insight or decision worth keeping for the long term.",
}

LABEL_COLORS = {
    "ephemeral": "#9CA3AF",
    "actionable": "#3B82F6",
    "archive": "#10B981",
}

if predict_clicked:
    text = (note_text or "").strip()
    if not text:
        st.warning("Please enter a note first.")
        st.stop()

    pred = model.predict([text])[0]
    probs = model.predict_proba([text])[0]
    classes = list(model.classes_)
    conf_map = {c: float(p) for c, p in zip(classes, probs)}

    color = LABEL_COLORS.get(pred, "#3B82F6")
    st.markdown(
        f"""
        <div style="padding:1rem;border-radius:8px;background:{color}22;
                    border-left:6px solid {color};margin:1rem 0;">
            <div style="font-size:0.85rem;color:#666;">Predicted label</div>
            <div style="font-size:1.6rem;font-weight:700;color:{color};">
                {pred.upper()}
            </div>
            <div style="margin-top:0.5rem;color:#444;">
                {LABEL_DESCRIPTIONS.get(pred, "")}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.subheader("Confidence by label")
    conf_df = pd.DataFrame({
        "label": list(conf_map.keys()),
        "confidence": list(conf_map.values()),
    }).sort_values("confidence", ascending=False)
    st.bar_chart(conf_df.set_index("label"))

    if use_llm:
        if not os.getenv("ANTHROPIC_API_KEY"):
            st.info("Set ANTHROPIC_API_KEY in your environment to enable explanations.")
        else:
            with st.spinner("Asking the LLM to explain..."):
                explanation = explain(text, pred, conf_map)
            st.subheader("Why this label?")
            st.write(explanation)


st.markdown("---")
st.caption(
    "BUSS305 Final Project. TF-IDF + Logistic Regression. "
    "Dataset synthesized via LLM data augmentation (Lecture 6)."
)
