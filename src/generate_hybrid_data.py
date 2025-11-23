"""
generate_hybrid_data.py
=======================

This script implements a *hybrid* data generation strategy for PII entity
recognition.  It accepts a file of real conversational transcripts and
injects synthetic PII entities into those transcripts.  The resulting data
contains real linguistic noise (from the base transcripts) and injected PII
values in STT style (spoken digits, "at" and "dot" for emails, etc.).

Usage example:

```
python src/generate_hybrid_data.py \
  --base_file data/base_transcripts.txt \
  --train_out data/hybrid_train.jsonl \
  --dev_out data/hybrid_dev.jsonl \
  --test_out data/hybrid_test.jsonl \
  --num_train 500 \
  --num_dev 100 \
  --num_test 100 \
  --seed 42
```

This will read lines from `base_transcripts.txt`, generate 500 training
examples, 100 development examples, and 100 test examples, and write them into
JSONL files.

Each output record has the same format as in generate_data.py: an `id`, the
full `text` with injected PII, and a list of `entities` with character
offsets and labels.

If the number of requested examples exceeds the number of lines in the base
file, the script will sample with replacement.

Author: OpenAI assistant
"""

import argparse
import json
import os
import random
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

# Ensure that the src directory is on the module search path so that we can
# import generate_data when this script is executed from the project root.
CURRENT_DIR = Path(__file__).resolve().parent
PARENT_DIR = CURRENT_DIR.parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))
if str(PARENT_DIR) not in sys.path:
    sys.path.insert(0, str(PARENT_DIR))

# We reuse helpers and configuration from generate_data.  Because we modified
# sys.path above, we can import generate_data directly regardless of the
# working directory.  These imports pull in constants and functions used to
# construct PII values and filler noise.
from generate_data import (
    FILLERS,
    FIRST_NAMES,
    LAST_NAMES,
    CITIES,
    LOCATIONS,
    generate_credit_card,
    generate_phone,
    generate_email,
    generate_date,
    ENTITY_TYPES,
)


def maybe_insert_filler(rng: random.Random) -> str:
    """Insert a conversational filler with 25% probability."""
    return rng.choice(FILLERS) if rng.random() < 0.25 else ""


def generate_person_name(rng: random.Random) -> str:
    """Sample a simple first/last name combination."""
    return f"{rng.choice(FIRST_NAMES)} {rng.choice(LAST_NAMES)}"


def pick_city(rng: random.Random) -> str:
    """Sample a city from the shared vocabulary."""
    return rng.choice(CITIES)


def pick_location(rng: random.Random) -> str:
    """Sample a location from the shared vocabulary."""
    return rng.choice(LOCATIONS)


def clean_transcript(text: str) -> str:
    """Lower-case and remove most punctuation from the base transcript.

    We strip trailing periods and commas and collapse multiple spaces.  This
    approximates the unpunctuated output of an ASR system.
    """
    # Lower-case
    text = text.strip().lower()
    # Replace punctuation with spaces
    text = re.sub(r"[^a-z0-9 ]", " ", text)
    # Collapse multiple spaces
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def build_hybrid_example(
    base_text: str,
    example_index: int,
    rng: random.Random,
) -> Tuple[str, List[Dict[str, object]]]:
    """Inject synthetic PII into a base transcript.

    For each example we select between one and three entity types to inject.
    We build the final text by taking the cleaned base transcript and appending
    or inserting synthetic PII phrases.  For simplicity and to ensure span
    computation is correct, this implementation appends the PII at the end of
    the transcript separated by filler words.

    Args:
        base_text: A cleaned transcript string (lower-case, punctuation removed).
        example_index: Index of this example (for ID generation if needed).
        rng: Random number generator for reproducibility.

    Returns:
        (final_text, entities) where final_text is the transcript with PII
        inserted and entities is a list of span dictionaries.
    """
    # Start with the base text
    parts: List[str] = [base_text]
    entities: List[Dict[str, object]] = []
    cursor = len(base_text)

    # Ensure a space between base text and injections
    if not base_text.endswith(" "):
        parts.append(" ")
        cursor += 1

    # Decide how many entity types to inject (1–3)
    num_injections = rng.randint(1, 3)
    chosen_types = rng.sample(list(ENTITY_TYPES), k=num_injections)

    for etype in chosen_types:
        # Optionally insert a filler before each injection
        filler = maybe_insert_filler(rng)
        if filler:
            parts.append(filler)
            parts.append(" ")
            cursor += len(filler) + 1
        # Generate the PII entity and its surrounding context
        if etype == "CREDIT_CARD":
            stt = generate_credit_card(rng)
            context_before = rng.choice([
                "my credit card number is",
                "credit card number",
                "the card is",
            ])
        elif etype == "PHONE":
            stt = generate_phone(rng)
            context_before = rng.choice([
                "my phone number is",
                "phone is",
                "call me at",
            ])
        elif etype == "EMAIL":
            stt = generate_email(rng)
            context_before = rng.choice([
                "my email is",
                "email is",
                "send it to",
            ])
        elif etype == "PERSON_NAME":
            name = generate_person_name(rng)
            stt = name
            context_before = rng.choice([
                "my name is",
                "this is",
                "i am",
            ])
        elif etype == "DATE":
            date_str = generate_date(rng)
            stt = date_str
            context_before = rng.choice([
                "on",
                "dated",
                "date is",
            ])
        elif etype == "CITY":
            city = pick_city(rng)
            stt = city
            context_before = rng.choice([
                "i am from",
                "living in",
                "based in",
            ])
        elif etype == "LOCATION":
            location = pick_location(rng)
            stt = location
            context_before = rng.choice([
                "meet me at",
                "address is",
                "at",
            ])
        else:
            # Skip unknown types
            continue
        # Append context and entity to text
        # Add context phrase
        parts.append(context_before)
        parts.append(" ")
        cursor += len(context_before) + 1
        # Record the start index for the entity
        start_idx = cursor
        parts.append(stt)
        cursor += len(stt)
        entities.append({"start": start_idx, "end": start_idx + len(stt), "label": etype})
        # Add a trailing space for readability
        parts.append(" ")
        cursor += 1

    final_text = "".join(parts).strip()
    return final_text, entities


def generate_hybrid_dataset(
    base_lines: Sequence[str],
    num_examples: int,
    rng: random.Random,
) -> List[Dict[str, object]]:
    """Generate a dataset by injecting synthetic PII into base transcripts."""
    dataset: List[Dict[str, object]] = []
    n_base = len(base_lines)
    for i in range(num_examples):
        base = base_lines[rng.randrange(n_base)]
        cleaned_base = clean_transcript(base)
        text, ents = build_hybrid_example(cleaned_base, i, rng)
        # Ensure at least one entity present
        if not ents:
            text, ents = build_hybrid_example(cleaned_base, i, rng)
        dataset.append({"id": f"utt_{i:06d}", "text": text, "entities": ents})
    return dataset


def save_jsonl(path: Path, data: Sequence[Dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for record in data:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate hybrid PII NER data from real transcripts")
    parser.add_argument("--base_file", type=str, required=True, help="Path to text file containing base transcripts (one per line)")
    parser.add_argument("--train_out", type=str, default="data/hybrid_train.jsonl", help="Path to output training data JSONL")
    parser.add_argument("--dev_out", type=str, default="data/hybrid_dev.jsonl", help="Path to output development data JSONL")
    parser.add_argument("--test_out", type=str, default="data/hybrid_test.jsonl", help="Path to output test data JSONL (optional)")
    parser.add_argument("--num_train", type=int, default=500, help="Number of training examples to generate")
    parser.add_argument("--num_dev", type=int, default=100, help="Number of development examples to generate")
    parser.add_argument("--num_test", type=int, default=0, help="Number of test examples to generate (0 to skip)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    base_path = Path(args.base_file)
    if not base_path.exists():
        raise FileNotFoundError(f"Base transcript file not found: {base_path}")
    base_lines = [line.strip() for line in base_path.open("r", encoding="utf-8") if line.strip()]
    if not base_lines:
        raise ValueError("Base transcript file is empty or all lines are blank")

    train_data = generate_hybrid_dataset(base_lines, args.num_train, rng)
    dev_data = generate_hybrid_dataset(base_lines, args.num_dev, rng)
    test_data: Optional[List[Dict[str, object]]] = None
    if args.num_test > 0:
        test_data = generate_hybrid_dataset(base_lines, args.num_test, rng)

    train_path = Path(args.train_out)
    dev_path = Path(args.dev_out)
    test_path = Path(args.test_out) if args.num_test > 0 else None
    train_path.parent.mkdir(parents=True, exist_ok=True)
    dev_path.parent.mkdir(parents=True, exist_ok=True)
    if test_path:
        test_path.parent.mkdir(parents=True, exist_ok=True)
    save_jsonl(train_path, train_data)
    save_jsonl(dev_path, dev_data)
    if test_path and test_data is not None:
        save_jsonl(test_path, test_data)

    print(f"Wrote {len(train_data)} training examples to {train_path}")
    print(f"Wrote {len(dev_data)} development examples to {dev_path}")
    if test_path and test_data is not None:
        print(f"Wrote {len(test_data)} test examples to {test_path}")


if __name__ == "__main__":
    main()
