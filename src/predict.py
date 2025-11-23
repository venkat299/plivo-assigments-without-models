import argparse
import json
import os

import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification

from labels import ID2LABEL, LABEL2ID, label_is_pii


def bio_to_spans(text, offsets, label_ids, attention_mask=None):
    spans = []
    current_label = None
    current_start = None
    current_end = None

    for idx, ((start, end), lid) in enumerate(zip(offsets, label_ids)):
        if attention_mask is not None and attention_mask[idx] == 0:
            continue
        if start == 0 and end == 0:  # special tokens / pads
            continue
        label = ID2LABEL.get(int(lid), "O")
        if label == "O":
            if current_label is not None:
                spans.append((current_start, current_end, current_label))
                current_label = None
            continue

        prefix, ent_type = label.split("-", 1)
        if prefix == "B":
            if current_label is not None:
                spans.append((current_start, current_end, current_label))
            current_label = ent_type
            current_start = start
            current_end = end
        elif prefix == "I":
            if current_label == ent_type:
                current_end = end
            else:
                if current_label is not None:
                    spans.append((current_start, current_end, current_label))
                current_label = ent_type
                current_start = start
                current_end = end

    if current_label is not None:
        spans.append((current_start, current_end, current_label))

    return spans


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", default="out")
    ap.add_argument("--model_name", default=None)
    ap.add_argument("--input", default="data/dev.jsonl")
    ap.add_argument("--output", default="out/dev_pred.json")
    ap.add_argument("--max_length", type=int, default=256)
    ap.add_argument(
        "--prob_threshold",
        type=float,
        default=0.7,
        help="Global prob threshold; use per-label overrides to customize.",
    )
    ap.add_argument(
        "--label_thresholds",
        type=str,
        default="",
        help="Per-label thresholds, e.g., 'CREDIT_CARD:0.9,PHONE:0.85,EMAIL:0.85'.",
    )
    ap.add_argument("--card_min_digits", type=int, default=12, help="Min digits to keep CREDIT_CARD span.")
    ap.add_argument("--phone_min_digits", type=int, default=8, help="Min digits to keep PHONE span.")
    ap.add_argument(
        "--email_require_atdot",
        action="store_true",
        help="Drop EMAIL spans missing both 'at' and 'dot' tokens.",
    )
    ap.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_dir if args.model_name is None else args.model_name)
    model = AutoModelForTokenClassification.from_pretrained(args.model_dir)
    model.to(args.device)
    model.eval()

    label_thresh = {}
    if args.label_thresholds:
        for kv in args.label_thresholds.split(","):
            if not kv:
                continue
            lab, val = kv.split(":")
            label_thresh[lab.strip()] = float(val)

    results = {}

    with open(args.input, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            text = obj["text"]
            uid = obj["id"]

            enc = tokenizer(
                text,
                return_offsets_mapping=True,
                truncation=True,
                max_length=args.max_length,
                return_tensors="pt",
            )
            offsets = enc["offset_mapping"][0].tolist()
            input_ids = enc["input_ids"].to(args.device)
            attention_mask = enc["attention_mask"].to(args.device)

            with torch.no_grad():
                out = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = out.logits[0]
                probs = torch.softmax(logits, dim=-1)
                max_probs, pred_ids = torch.max(probs, dim=-1)

                pred_ids = pred_ids.cpu().tolist()
                max_probs = max_probs.cpu().tolist()

                # Precision guard: demote low-confidence non-O predictions to O
                for i, (pid, prob) in enumerate(zip(pred_ids, max_probs)):
                    if pid == LABEL2ID["O"]:
                        continue
                    lab = ID2LABEL[int(pid)].split("-", 1)[-1]
                    thresh = label_thresh.get(lab, args.prob_threshold)
                    if prob < thresh:
                        pred_ids[i] = LABEL2ID["O"]

            spans = bio_to_spans(text, offsets, pred_ids, attention_mask=attention_mask[0].tolist())

            DIGIT_WORDS = {
                "zero",
                "one",
                "two",
                "three",
                "four",
                "five",
                "six",
                "seven",
                "eight",
                "nine",
                "oh",
                "o",
                "double",
                "triple",
            }

            def numberish_count(s: str) -> int:
                toks = s.lower().split()
                count = 0
                for tok in toks:
                    if tok.isdigit():
                        count += len(tok)
                    elif tok in {"double", "triple"}:
                        count += 2 if tok == "double" else 3
                    elif tok in DIGIT_WORDS:
                        count += 1
                return count

            def looks_like_email(s: str) -> bool:
                toks = s.lower().split()
                return ("at" in toks) and ("dot" in toks)

            ents = []
            for s, e, lab in spans:
                span_text = text[s:e]
                if lab == "CREDIT_CARD" and numberish_count(span_text) < args.card_min_digits:
                    continue
                if lab == "PHONE" and numberish_count(span_text) < args.phone_min_digits:
                    continue
                if lab == "EMAIL" and args.email_require_atdot and not looks_like_email(span_text):
                    continue

                ents.append(
                    {
                        "start": int(s),
                        "end": int(e),
                        "label": lab,
                        "pii": bool(label_is_pii(lab)),
                    }
                )
            results[uid] = ents

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"Wrote predictions for {len(results)} utterances to {args.output}")


if __name__ == "__main__":
    main()
