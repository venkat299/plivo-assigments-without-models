"""
Utility to synthesize noisy STT-style NER data for PII detection.
Produces JSONL files with char-level spans for entities.
"""
import argparse
import json
import os
import random
from typing import Dict, List

DIGIT_WORDS = {
    "0": "zero",
    "1": "one",
    "2": "two",
    "3": "three",
    "4": "four",
    "5": "five",
    "6": "six",
    "7": "seven",
    "8": "eight",
    "9": "nine",
}

FIRST_NAMES = [
    "ramesh",
    "priyanka",
    "rohan",
    "megha",
    "amit",
    "fatima",
    "sneha",
    "arjun",
    "kavya",
    "rehan",
    "neha",
    "vikram",
    "tanya",
    "varun",
    "nina",
]

LAST_NAMES = [
    "sharma",
    "verma",
    "mehta",
    "khan",
    "iyer",
    "rao",
    "patel",
    "das",
    "singh",
    "nair",
    "agarwal",
    "joshi",
    "reddy",
    "gupta",
    "sehgal",
]

DOMAINS = [
    "gmail dot com",
    "yahoo dot com",
    "hotmail dot com",
    "outlook dot com",
    "proton mail dot com",
    "icloud dot com",
    "pm dot me",
]

CITIES = [
    "mumbai",
    "delhi",
    "chennai",
    "pune",
    "bangalore",
    "hyderabad",
    "kolkata",
    "jaipur",
    "ahmedabad",
    "kochi",
]

LOCATIONS = [
    "terminal three",
    "central mall",
    "city center",
    "bus stand",
    "railway station",
    "north gate",
    "airport road",
    "tech park",
    "main square",
    "sector five",
]

FILLER = ["um", "please", "okay", "ya", "right", "you know", "like"]


class ExampleBuilder:
    """Helper to build text and keep span offsets."""

    def __init__(self):
        self.text_parts: List[str] = []
        self.entities: List[Dict[str, object]] = []

    @property
    def text(self) -> str:
        return " ".join(self.text_parts)

    def add_chunk(self, chunk: str, label: str = None):
        chunk = chunk.strip()
        if not chunk:
            return
        start = len(self.text)
        if self.text_parts:
            start += 1  # account for joining space
        end = start + len(chunk)
        self.text_parts.append(chunk)
        if label:
            self.entities.append({"start": start, "end": end, "label": label})


def digits_to_words(seq: str, allow_double: bool = True) -> str:
    """Convert digit string to STT-style words, optionally using 'double'."""
    words: List[str] = []
    i = 0
    while i < len(seq):
        if (
            allow_double
            and i + 1 < len(seq)
            and seq[i] == seq[i + 1]
            and random.random() < 0.25
        ):
            words.append("double")
            words.append(DIGIT_WORDS[seq[i]])
            i += 2
        else:
            words.append(DIGIT_WORDS[seq[i]])
            i += 1
    return " ".join(words)


def random_credit_card() -> str:
    digits = "".join(random.choice("0123456789") for _ in range(16))
    if random.random() < 0.5:
        groups = [digits[i : i + 4] for i in range(0, 16, 4)]
        return " ".join(groups)
    return digits_to_words(digits)


def random_phone() -> str:
    base = "".join(random.choice("0123456789") for _ in range(10))
    if random.random() < 0.5:
        return digits_to_words(base)
    if random.random() < 0.3:
        return "+91 " + " ".join([base[:5], base[5:]])
    return " ".join([base[:3], base[3:6], base[6:]])


def random_name() -> str:
    first = random.choice(FIRST_NAMES)
    last = random.choice(LAST_NAMES)
    if random.random() < 0.2:
        middle = random.choice(FIRST_NAMES)
        return f"{first} {middle} {last}"
    return f"{first} {last}"


def random_email(name: str) -> str:
    user = name.replace(" ", " dot ")
    if random.random() < 0.3:
        user = user + " " + digits_to_words(str(random.randint(1, 99)))
    domain = random.choice(DOMAINS)
    return f"{user} at {domain}"


def random_date() -> str:
    day = random.randint(1, 28)
    year = random.randint(2023, 2027)
    month_words = [
        "january",
        "february",
        "march",
        "april",
        "may",
        "june",
        "july",
        "august",
        "september",
        "october",
        "november",
        "december",
    ]
    if random.random() < 0.4:
        return f"{day:02d} {random.choice(month_words)} {year}"
    return f"{digits_to_words(str(day).zfill(2))} {random.choice(month_words)} {digits_to_words(str(year))}"


def maybe_fill(builder: ExampleBuilder, prob: float = 0.4):
    if random.random() < prob:
        filler = random.choice(FILLER)
        builder.add_chunk(filler)


def build_example(uid: str) -> Dict[str, object]:
    b = ExampleBuilder()

    scenario = random.choice(
        [
            "card_email_name",
            "phone_city_date",
            "name_phone_only",
            "location_city",
            "card_only",
            "email_only",
        ]
    )

    name = random_name()
    email = random_email(name)
    card = random_credit_card()
    phone = random_phone()
    city = random.choice(CITIES)
    location = random.choice(LOCATIONS)
    date = random_date()

    maybe_fill(b)

    if scenario == "card_email_name":
        b.add_chunk("my credit card number is")
        b.add_chunk(card, label="CREDIT_CARD")
        maybe_fill(b, prob=0.3)
        b.add_chunk("card holder name is")
        b.add_chunk(name, label="PERSON_NAME")
        b.add_chunk("and email is")
        b.add_chunk(email, label="EMAIL")
    elif scenario == "phone_city_date":
        b.add_chunk("call me on")
        b.add_chunk(phone, label="PHONE")
        maybe_fill(b, prob=0.2)
        b.add_chunk("i am calling from")
        b.add_chunk(city, label="CITY")
        b.add_chunk("and traveling on")
        b.add_chunk(date, label="DATE")
    elif scenario == "name_phone_only":
        b.add_chunk("this is")
        b.add_chunk(name, label="PERSON_NAME")
        b.add_chunk("my number is")
        b.add_chunk(phone, label="PHONE")
    elif scenario == "location_city":
        b.add_chunk("meet me at")
        b.add_chunk(location, label="LOCATION")
        b.add_chunk("in")
        b.add_chunk(city, label="CITY")
    elif scenario == "card_only":
        b.add_chunk("note my card is")
        b.add_chunk(card, label="CREDIT_CARD")
        b.add_chunk("it expires on")
        b.add_chunk(date, label="DATE")
    elif scenario == "email_only":
        b.add_chunk("email id is")
        b.add_chunk(email, label="EMAIL")
        if random.random() < 0.6:
            b.add_chunk("name")
            b.add_chunk(name, label="PERSON_NAME")
    else:
        b.add_chunk("call me on")
        b.add_chunk(phone, label="PHONE")

    maybe_fill(b, prob=0.2)

    return {"id": uid, "text": b.text, "entities": b.entities}


def write_jsonl(path: str, items: List[Dict[str, object]]):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for obj in items:
            f.write(json.dumps(obj, ensure_ascii=False))
            f.write("\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_out", default="data/train.jsonl")
    ap.add_argument("--dev_out", default="data/dev.jsonl")
    ap.add_argument("--train_size", type=int, default=900)
    ap.add_argument("--dev_size", type=int, default=150)
    ap.add_argument("--seed", type=int, default=42, help="RNG seed for train generation")
    ap.add_argument(
        "--dev_seed",
        type=int,
        default=None,
        help="Optional separate seed for dev generation (defaults to seed+1)",
    )
    args = ap.parse_args()

    random.seed(args.seed)
    train = [build_example(f"utt_train_{i:04d}") for i in range(args.train_size)]

    dev_seed = args.dev_seed if args.dev_seed is not None else args.seed + 1
    random.seed(dev_seed)
    dev = [build_example(f"utt_dev_{i:04d}") for i in range(args.dev_size)]

    write_jsonl(args.train_out, train)
    write_jsonl(args.dev_out, dev)
    print(f"Wrote {len(train)} train examples to {args.train_out}")
    print(f"Wrote {len(dev)} dev examples to {args.dev_out} (dev_seed={dev_seed})")


if __name__ == "__main__":
    main()
