"""
generate_data.py
=================

This script synthesizes a realistic dataset for PII recognition from noisy
Speech-to-Text (STT) transcripts. It uses scenario-based templates, inserts
conversational disfluencies (um, uh, repetitions), and varies entity formatting
to closely mimic real-world ASR outputs.

Features:
  - Generates JSON Lines with 'id', 'text', and character-offset 'entities'.
  - Simulates Indian English context (names, locations).
  - Simulates ASR noise (hesitations, self-corrections).
  - Handles complex spoken entities (emails with "underscore", grouped phone numbers).

Usage:
  python generate_data.py --train_out data/train.jsonl --dev_out data/dev.jsonl \
      --num_train 1000 --num_dev 200 --seed 42
"""

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

# -----------------------------------------------------------------------------
# 1. Data & Vocabularies
# -----------------------------------------------------------------------------

# Conversational fillers and "garbage" words
FILLERS = ["um", "uh", "ah", "er", "like", "you know", "actually", "basically", "sort of", "i mean"]
# Words indicating a correction
CORRECTIONS = ["no sorry", "i mean", "correction", "wait"]

# Indian & Western Mix
FIRST_NAMES = (
    "aarav", "vihaan", "aditya", "sai", "arjun", "rohan", "priya", "diya", "ananya",
    "pooja", "neha", "sivakumar", "john", "david", "sarah", "jessica", "mohamed",
    "fatima", "karthik", "lakshmi"
)
LAST_NAMES = (
    "iyer", "reddy", "naidu", "patel", "sharma", "singh", "kumar", "gupta", "malhotra",
    "fernandes", "smith", "doe", "johnson", "khan", "ali", "nair", "menon", "rao"
)

CITIES = (
    "chennai", "mumbai", "bangalore", "delhi", "kolkata", "hyderabad", "pune",
    "kochi", "coimbatore", "madurai", "new york", "london", "dubai", "singapore"
)

# More specific address components
LOCATIONS = (
    "anna nagar", "t nagar", "velachery", "adyar", "mylapore", "whitefield",
    "connaught place", "church street", "marine drive", "central station",
    "airport road", "sector 45", "flat 302", "plot number 12"
)

EMAIL_DOMAINS = ("gmail", "yahoo", "outlook", "hotmail", "zoho", "corporate")
EMAIL_TLDS = ("com", "co dot in", "org", "net", "io")

DIGIT_TO_WORD = {
    "0": "zero", "1": "one", "2": "two", "3": "three", "4": "four",
    "5": "five", "6": "six", "7": "seven", "8": "eight", "9": "nine"
}

MONTHS = (
    "january", "february", "march", "april", "may", "june",
    "july", "august", "september", "october", "november", "december"
)

ENTITY_TYPES = ("CREDIT_CARD", "PHONE", "EMAIL", "PERSON_NAME", "DATE", "CITY", "LOCATION")

# -----------------------------------------------------------------------------
# 2. Entity Generation Logic
# -----------------------------------------------------------------------------


def text_to_spoken_digits(text: str, rng: random.Random) -> str:
    """Converts a string of digits to spoken words with variety (grouping, doubles)."""
    words: List[str] = []
    i = 0
    while i < len(text):
        char = text[i]
        if not char.isdigit():
            words.append(char)
            i += 1
            continue

        # 20% chance to say "double X" if two consecutive digits match
        if i + 1 < len(text) and text[i + 1] == char and rng.random() < 0.2:
            words.append(f"double {DIGIT_TO_WORD[char]}")
            i += 2
        # 15% chance to say "triple X"
        elif i + 2 < len(text) and text[i + 1] == char and text[i + 2] == char and rng.random() < 0.15:
            words.append(f"triple {DIGIT_TO_WORD[char]}")
            i += 3
        else:
            words.append(DIGIT_TO_WORD[char])
            i += 1

    return " ".join(words)


def generate_phone(rng: random.Random) -> str:
    """Generates a phone number spoken in groups (common in India)."""
    digits = "".join([str(rng.randint(0, 9)) for _ in range(10)])

    # Variety in grouping: "98400 12345" or "984 001 2345" or continuous
    style = rng.choice(["split_5_5", "split_3_3_4", "continuous"])

    if style == "split_5_5":
        part1 = text_to_spoken_digits(digits[:5], rng)
        part2 = text_to_spoken_digits(digits[5:], rng)
        return f"{part1} {part2}"
    if style == "split_3_3_4":
        part1 = text_to_spoken_digits(digits[:3], rng)
        part2 = text_to_spoken_digits(digits[3:6], rng)
        part3 = text_to_spoken_digits(digits[6:], rng)
        return f"{part1} {part2} {part3}"
    return text_to_spoken_digits(digits, rng)


def generate_email(rng: random.Random) -> str:
    """Generates an email with special char handling (underscore, dot)."""
    name = rng.choice(FIRST_NAMES)
    sep = rng.choice(["", ".", "_", str(rng.randint(1, 99))])
    last = rng.choice(LAST_NAMES) if rng.random() > 0.3 else ""
    domain = rng.choice(EMAIL_DOMAINS)
    tld = rng.choice(EMAIL_TLDS)

    parts = [name]
    if sep == ".":
        parts.append("dot")
    elif sep == "_":
        parts.append("underscore")
    elif sep.isdigit():
        parts.append(text_to_spoken_digits(sep, rng))

    if last:
        parts.append(last)

    user_part = " ".join(parts)
    domain_part = f"{domain} dot {tld}"

    return f"{user_part} at {domain_part}"


def generate_date(rng: random.Random) -> str:
    """Generates spoken dates in various formats."""
    day = rng.randint(1, 31)
    month = rng.choice(MONTHS)
    year = rng.randint(1980, 2025)

    fmt = rng.choice(["ordinal", "cardinal", "short"])

    if fmt == "ordinal":
        suffixes = {
            1: "first",
            2: "second",
            3: "third",
            21: "twenty first",
            22: "twenty second",
            23: "twenty third",
            31: "thirty first",
        }
        day_spoken = suffixes.get(day, text_to_spoken_digits(str(day), rng) + "th")
        return f"{day_spoken} of {month} {text_to_spoken_digits(str(year), rng)}"
    if fmt == "cardinal":
        return f"{month} {text_to_spoken_digits(str(day), rng)} {text_to_spoken_digits(str(year), rng)}"
    day_spoken = text_to_spoken_digits(str(day), rng)
    return f"{day_spoken} {month} {text_to_spoken_digits(str(year), rng)}"


def generate_credit_card(rng: random.Random) -> str:
    digits = "".join([str(rng.randint(0, 9)) for _ in range(16)])
    p1 = text_to_spoken_digits(digits[0:4], rng)
    p2 = text_to_spoken_digits(digits[4:8], rng)
    p3 = text_to_spoken_digits(digits[8:12], rng)
    p4 = text_to_spoken_digits(digits[12:16], rng)
    return f"{p1} {p2} {p3} {p4}"


# -----------------------------------------------------------------------------
# 3. Scenario Construction & Noise
# -----------------------------------------------------------------------------


def inject_noise(text: str, rng: random.Random) -> str:
    """Injects disfluencies and repetitions into the text."""
    words = text.split()
    noisy_words: List[str] = []

    for i, word in enumerate(words):
        # 5% chance to repeat the word (stutter)
        if rng.random() < 0.05 and len(word) > 2:
            noisy_words.append(word)

        # 5% chance to insert a filler before the word
        if rng.random() < 0.05:
            noisy_words.append(rng.choice(FILLERS))

        # 2% chance of self-correction
        if rng.random() < 0.02 and i > 0:
            noisy_words.append(rng.choice(CORRECTIONS))
            noisy_words.append(words[i - 1])

        noisy_words.append(word)

    return " ".join(noisy_words)


def build_scenario(rng: random.Random, dev_mode: bool = False) -> List[Tuple[str, Optional[str]]]:
    """
    Constructs a list of (text_segment, label) tuples based on logical templates.
    This ensures the entities appear in a natural sentence structure.
    """
    scenarios = ["banking", "delivery", "personal_intro", "complaint"]
    dev_only = ["support", "travel_change"]
    if dev_mode:
        scenarios += dev_only
    scenario = rng.choice(scenarios)
    segments: List[Tuple[str, Optional[str]]] = []

    if scenario == "banking":
        segments.append((rng.choice(["hi", "hello"]) + " i need to verify my account ", None))
        segments.append(("my card number is ", None))
        segments.append((generate_credit_card(rng), "CREDIT_CARD"))
        segments.append((" and the registered name is ", None))
        segments.append((f"{rng.choice(FIRST_NAMES)} {rng.choice(LAST_NAMES)}", "PERSON_NAME"))

    elif scenario == "delivery":
        segments.append(("please deliver the package to ", None))
        segments.append((f"{rng.choice(LOCATIONS)}", "LOCATION"))
        segments.append((" in ", None))
        segments.append((rng.choice(CITIES), "CITY"))
        segments.append((" you can contact me on ", None))
        segments.append((generate_phone(rng), "PHONE"))

    elif scenario == "personal_intro":
        segments.append(("my name is ", None))
        segments.append((f"{rng.choice(FIRST_NAMES)} {rng.choice(LAST_NAMES)}", "PERSON_NAME"))
        segments.append((" and i was born on ", None))
        segments.append((generate_date(rng), "DATE"))
        segments.append((" currently living in ", None))
        segments.append((rng.choice(CITIES), "CITY"))

    elif scenario == "complaint":
        segments.append(("i am writing regarding an issue with order from ", None))
        segments.append((generate_date(rng), "DATE"))
        segments.append((" please reply to my email ", None))
        segments.append((generate_email(rng), "EMAIL"))
        segments.append((" or call ", None))
        segments.append((generate_phone(rng), "PHONE"))

    elif scenario == "support":  # dev-only held-out layout
        segments.append(("support ticket for ", None))
        segments.append((rng.choice(["billing", "login", "delivery", "upgrade"]), None))
        segments.append((" user ", None))
        segments.append((f"{rng.choice(FIRST_NAMES)} {rng.choice(LAST_NAMES)}", "PERSON_NAME"))
        segments.append((" call back at ", None))
        segments.append((generate_phone(rng), "PHONE"))
        segments.append((" and city ", None))
        segments.append((rng.choice(CITIES), "CITY"))

    elif scenario == "travel_change":  # dev-only held-out layout
        segments.append(("i need to reschedule travel on ", None))
        segments.append((generate_date(rng), "DATE"))
        segments.append((" departing from ", None))
        segments.append((rng.choice(CITIES), "CITY"))
        segments.append((" meeting point ", None))
        segments.append((rng.choice(LOCATIONS), "LOCATION"))
        segments.append((" confirmation email ", None))
        segments.append((generate_email(rng), "EMAIL"))

    return segments


def dev_noise_segments(rng: random.Random) -> List[Tuple[str, Optional[str]]]:
    """Generate distractor segments to provoke false positives in dev."""
    segments: List[Tuple[str, Optional[str]]] = []

    # Fake phone-like number without label
    digits = "".join(str(rng.randint(0, 9)) for _ in range(rng.randint(8, 12)))
    phoneish = text_to_spoken_digits(digits, rng)
    segments.append((f" reach me maybe on {phoneish} ", None))

    # Partial email (missing tld)
    name = rng.choice(FIRST_NAMES)
    partial_email = f"{name} at {rng.choice(EMAIL_DOMAINS)}"
    segments.append((f" alternate contact {partial_email} ", None))

    # Card-like number without label
    cardish = " ".join(text_to_spoken_digits("".join(str(rng.randint(0, 9)) for _ in range(4)), rng) for _ in range(3))
    segments.append((f" reference number {cardish} ", None))

    rng.shuffle(segments)
    # Sample 1-2 noise snippets
    keep = rng.sample(segments, rng.randint(1, min(2, len(segments))))
    return keep


def build_example(idx: int, rng: random.Random, dev_mode: bool = False) -> Dict[str, object]:
    """Builds a full example with noise and index calculation."""
    raw_segments = build_scenario(rng, dev_mode=dev_mode)

    if dev_mode:
        noise = dev_noise_segments(rng)
        insert_at = rng.randint(0, len(raw_segments))
        raw_segments[insert_at:insert_at] = noise

    final_text_parts: List[str] = []
    entities: List[Dict[str, object]] = []
    current_cursor = 0

    for text, label in raw_segments:
        if label is None:
            noisy_text = inject_noise(text, rng)
            clean_part = noisy_text.strip() + " "
            final_text_parts.append(clean_part)
            current_cursor += len(clean_part)
        else:
            val = text.strip()
            start = current_cursor
            end = start + len(val)
            final_text_parts.append(val + " ")
            entities.append({"start": start, "end": end, "label": label})
            current_cursor += len(val) + 1

    full_text = "".join(final_text_parts).strip()

    return {
        "id": f"utt_{idx:06d}",
        "text": full_text,
        "entities": entities,
    }


# -----------------------------------------------------------------------------
# 4. Main Execution
# -----------------------------------------------------------------------------


def generate_dataset(num_examples: int, seed: int, dev_mode: bool = False) -> List[Dict[str, object]]:
    rng = random.Random(seed)
    data = []
    for i in range(num_examples):
        data.append(build_example(i, rng, dev_mode=dev_mode))
    return data


def save_jsonl(path: Path, data: Sequence[Dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for record in data:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Enhanced Synthetic PII Generator")
    parser.add_argument("--train_out", type=str, default="data/train.jsonl")
    parser.add_argument("--dev_out", type=str, default="data/dev.jsonl")
    parser.add_argument("--test_out", type=str, default=None, help="Optional test JSONL output path.")
    parser.add_argument("--num_train", type=int, default=800)
    parser.add_argument("--num_dev", type=int, default=150)
    parser.add_argument("--num_test", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dev_seed", type=int, default=None, help="Optional separate seed for dev")
    parser.add_argument("--test_seed", type=int, default=None, help="Optional separate seed for test")
    parser.add_argument("--train_dev_mode", action="store_true", help="Use dev-mode noise for train generation")
    parser.add_argument("--test_dev_mode", action="store_true", help="Use dev-mode noise for test generation")
    args = parser.parse_args()

    train_path = Path(args.train_out)
    dev_path = Path(args.dev_out)
    train_path.parent.mkdir(parents=True, exist_ok=True)
    dev_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Generating {args.num_train} training examples (dev_mode={args.train_dev_mode})...")
    train_data = generate_dataset(args.num_train, args.seed, dev_mode=args.train_dev_mode)

    dev_seed = args.dev_seed if args.dev_seed is not None else args.seed + 1
    print(f"Generating {args.num_dev} dev examples with extra noise (seed={dev_seed})...")
    dev_data = generate_dataset(args.num_dev, dev_seed, dev_mode=True)

    save_jsonl(train_path, train_data)
    save_jsonl(dev_path, dev_data)

    if args.test_out:
        test_seed = args.test_seed if args.test_seed is not None else args.seed + 2
        print(f"Generating {args.num_test} test examples (dev_mode={args.test_dev_mode}, seed={test_seed})...")
        test_data = generate_dataset(args.num_test, test_seed, dev_mode=args.test_dev_mode)
        test_path = Path(args.test_out)
        test_path.parent.mkdir(parents=True, exist_ok=True)
        save_jsonl(test_path, test_data)

    print("Done.")


if __name__ == "__main__":
    main()
