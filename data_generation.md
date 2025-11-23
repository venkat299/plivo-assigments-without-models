# Data Generation Overview

## Datasets/Folders
- `data/` – Synthetic STT-style data from `src/generate_data.py`.
- `data_hybrid/` – Hybrid data (real transcripts + injected PII) from `src/generate_hybrid_data.py`.
- `data_mix/` – Mixed setup: synthetic train + hybrid dev/test; includes a hard-negative train variant.

## Synthetic (`src/generate_data.py`)
- Templates: banking, delivery, personal_intro, complaint; dev-mode adds support, travel_change + distractor near-misses (phone-like numbers, partial emails, card-ish refs).
- Noise: fillers, stutter, self-correction; spoken digits, “at/dot” emails.
- Flags:
  - `dev_mode=False` for standard; `dev_mode=True` injects extra scenarios + near-miss noise.
  - CLI supports separate seeds for train/dev/test and optional dev-mode for train/test.
- Current snapshot: `data/train.jsonl`, `data/dev.jsonl`, `data/test.jsonl` generated with `dev_mode=True` for all splits (seeds 42/123/321), sizes 1000/200/200; used directly as `data_mix/train.jsonl`.
- Example generation (hard-mode for all splits):
  ```
  .venv/bin/python src/generate_data.py \
    --train_out data/train.jsonl --dev_out data/dev.jsonl --test_out data/test.jsonl \
    --num_train 1000 --num_dev 200 --num_test 200 \
    --seed 42 --dev_seed 123 --test_seed 321 \
    --train_dev_mode --test_dev_mode
  ```

## Hybrid (`src/generate_hybrid_data.py`)
- Base transcripts: real conversational lines (`src/base_transcripts.txt` or other).
- Process: clean transcript (lowercase, strip punctuation), then append injected PII (credit cards, phones, emails, names, dates, cities, locations) with context phrases and optional fillers.
- Uses generators from `generate_data.py` for PII strings (spoken digits, “at/dot” emails).
- Example:
  ```
  .venv/bin/python src/generate_hybrid_data.py \
    --base_file src/base_transcripts.txt \
    --train_out data_hybrid/train.jsonl \
    --dev_out data_hybrid/dev.jsonl \
    --num_train 1000 --num_dev 200 --seed 42
  ```

## Mixed (`data_mix/*`)
- Composition: `train.jsonl` is a copy of the synthetic hard-mode train (`data/train.jsonl`); `dev.jsonl`/`test.jsonl` are the hybrid splits (`data_hybrid/dev.jsonl`, `data_hybrid/test.jsonl`).
- Hard negatives: `train_hardneg.jsonl` = synthetic train (1000) + 200 unlabeled “neg_XXXX” examples containing PII-like spans left unlabeled to stress precision and decoding filters.
- Rebuild:
  ```
  cp data/train.jsonl data_mix/train.jsonl
  cp data_hybrid/dev.jsonl data_mix/dev.jsonl
  cp data_hybrid/test.jsonl data_mix/test.jsonl
  # For hard negatives reuse the committed data_mix/train_hardneg.jsonl (contains appended neg_0000–neg_0199 block).
  ```

## Differences
- **Source text**: `data/` is fully synthetic; `data_hybrid/` preserves real conversational text and injects PII.
- **Noise**: Synthetic dev-mode adds adversarial near-misses; hybrid relies on real transcript noise + injected PII, no near-miss injection unless added to generator.
- **Labels**: Both emit labeled JSONL with spans; test in synthetic generation currently labeled (use for eval or strip for inference-only).

## Tracking
- Record seeds, dev-mode flags, sizes, and base transcript files used for each dataset version in `metrics.md`/experiment notes. Note when mixing splits (e.g., data_mix uses synthetic train + hybrid dev/test; train_hardneg adds unlabeled neg_XXXX rows).
