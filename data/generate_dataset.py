"""
generate_dataset.py
-------------------
Synthesize a labeled dataset of personal notes for memory lifecycle classification.

Lecture 6 (LLM Data Augmentation) techniques used:
- Label-conditioned prompting: each prompt specifies the target label
- Controllable attributes: vary voice, length, domain to push diversity
- Post-hoc deduplication: remove near-duplicate notes
- Seed-anchored variation: each batch starts from a domain and voice hint

Output: data/notes.csv with columns [text, label]
Labels: ephemeral | actionable | archive
"""

import os
import csv
import time
import random
from pathlib import Path
from dotenv import load_dotenv
from anthropic import Anthropic

load_dotenv()

client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
MODEL = os.getenv("SYNTHESIS_MODEL", "claude-haiku-4-5-20251001")

# Target counts per label (balanced classes)
TARGETS = {
    "ephemeral": 700,
    "actionable": 700,
    "archive": 700,
}

# Diversity knobs (Lecture 6: controllable attributes)
DOMAINS = [
    "software engineering work", "personal life errands", "academic studies",
    "business meetings", "creative writing ideas", "fitness and health tracking",
    "financial planning", "reading and learning notes",
]

VOICES = ["terse", "casual", "formal", "stream-of-consciousness", "bullet-point"]

LABEL_DEFINITIONS = {
    "ephemeral": (
        "Short-lived reminders or thoughts that lose value within hours or days. "
        "Examples: 'Pick up milk', 'Reply to John later', 'Remember to close the window', "
        "'Need coffee', 'That meeting was boring'. These notes have no long-term value."
    ),
    "actionable": (
        "Specific tasks with clear next steps, often with deadlines or owners. "
        "Examples: 'Fix login bug by Friday', 'Email professor about extension', "
        "'Submit tax form before April 15', 'Refactor auth module this sprint'. "
        "These notes drive concrete action within a defined timeframe."
    ),
    "archive": (
        "Insights, lessons, decisions, or knowledge worth preserving long-term. "
        "Examples: 'Key insight: caching layer cut latency 40 percent', "
        "'Decision rationale: chose Postgres over Mongo because of joins', "
        "'Book takeaway: deep work requires uninterrupted blocks'. "
        "These notes hold lasting value as reference material."
    ),
}

BATCH_SIZE = 20  # notes per LLM call


def build_prompt(label: str, domain: str, voice: str, n: int) -> str:
    definition = LABEL_DEFINITIONS[label]
    return f"""Generate {n} realistic personal notes that a knowledge worker might jot down.

LABEL: {label}
DEFINITION: {definition}

CONTEXT DOMAIN: {domain}
WRITING VOICE: {voice}

REQUIREMENTS:
- Each note is 1 to 3 sentences (real notes, not essays).
- Notes must clearly belong to the "{label}" category as defined above.
- Notes should sound like something a real person would actually write to themselves.
- Vary topics within the domain. Avoid repeating the same phrasing.
- No numbering, no bullets, no quotes. Just plain text, one note per line.

Output exactly {n} notes, separated by newlines. No extra commentary."""


def generate_batch(label: str, domain: str, voice: str, n: int) -> list:
    prompt = build_prompt(label, domain, voice, n)
    try:
        resp = client.messages.create(
            model=MODEL,
            max_tokens=1500,
            temperature=0.9,
            system="You are a dataset generator producing realistic personal notes.",
            messages=[{"role": "user", "content": prompt}],
        )
        text = resp.content[0].text if resp.content else ""
        notes = [line.strip().lstrip("-*0123456789. ") for line in text.split("\n")]
        notes = [n for n in notes if 10 <= len(n) <= 500]
        return notes
    except Exception as e:
        print(f"  batch failed: {e}")
        return []


def main():
    out_path = Path("data/notes.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    all_rows: list[tuple] = []
    seen_texts: set = set()

    for label, target in TARGETS.items():
        print(f"\nGenerating {target} notes for label='{label}'...")
        collected = 0
        attempts = 0
        max_attempts = target * 3

        while collected < target and attempts < max_attempts:
            domain = random.choice(DOMAINS)
            voice = random.choice(VOICES)
            batch = generate_batch(label, domain, voice, BATCH_SIZE)

            for note in batch:
                key = note.lower().strip()
                if key in seen_texts:
                    continue
                seen_texts.add(key)
                all_rows.append((note, label))
                collected += 1
                if collected >= target:
                    break

            attempts += BATCH_SIZE
            print(f"  progress: {collected}/{target}")
            time.sleep(0.3)

    random.shuffle(all_rows)

    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["text", "label"])
        writer.writerows(all_rows)

    print(f"\nDone. Wrote {len(all_rows)} notes to {out_path}")
    print("Label distribution:")
    for label in TARGETS:
        count = sum(1 for _, l in all_rows if l == label)
        print(f"  {label}: {count}")


if __name__ == "__main__":
    main()
