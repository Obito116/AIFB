"""
llm_explain.py
--------------
Optional layer: ask an LLM to explain a classifier prediction in plain language.

This is the Advanced Track addition: the ML model makes the prediction,
the LLM adds an interpretive explanation. The classifier remains the source
of truth, the LLM is constrained to explain, not to overrule.
"""

import os
from dotenv import load_dotenv
from anthropic import Anthropic

load_dotenv()

_client = None


def get_client() -> Anthropic:
    global _client
    if _client is None:
        _client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    return _client


def explain(note: str, predicted_label: str, confidences: dict) -> str:
    """
    Return a short natural-language explanation for why the note got this label.

    Constraints (Lecture 6: constrained edits, label-conditioned):
    - The LLM is told the predicted label and must justify it.
    - The LLM may not change the label.
    """
    model = os.getenv("EXPLAIN_MODEL", "claude-sonnet-4-6")
    conf_str = ", ".join(f"{k}: {v:.2f}" for k, v in confidences.items())

    system = (
        "You are an assistant that explains classification decisions. "
        "You will be given a personal note, a predicted label, and the model's "
        "confidence scores. Your job is to write a 2-3 sentence plain-language "
        "explanation for why this label fits. Do not contradict the label."
    )
    user = (
        f"NOTE:\n{note}\n\n"
        f"PREDICTED LABEL: {predicted_label}\n"
        f"CONFIDENCES: {conf_str}\n\n"
        f"Write a 2-3 sentence explanation."
    )

    try:
        resp = get_client().messages.create(
            model=model,
            max_tokens=200,
            temperature=0.4,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        return resp.content[0].text if resp.content else ""
    except Exception as e:
        return f"(LLM explanation unavailable: {e})"
