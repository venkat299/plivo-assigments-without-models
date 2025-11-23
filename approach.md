# Approach Overview

## Objective and Constraints
- Build a PII NER model for noisy STT transcripts with strong PII precision under a CPU p95 latency target of ~20 ms (batch size 1). Entity types: CREDIT_CARD, PHONE, EMAIL, PERSON_NAME, DATE, CITY, LOCATION; PII flag true for the first five.
- Keep experimentation reproducible via tracked data generation, prediction artifacts, and metrics tables in `metrics.md` and experiment-specific files.

## Data Strategy
- **Synthetic hard-mode (`data/*`)**: Regenerated with `src/generate_data.py` using `dev_mode=True` for train/dev/test (seeds 42/123/321, sizes 1000/200/200). Adds held-out templates (support, travel_change) plus near-miss noise (phone-ish numbers, partial emails, card-like refs) to pressure precision.
- **Hybrid real+synthetic (`data_hybrid/*`)**: `src/generate_hybrid_data.py` cleans real transcripts from `src/base_transcripts.txt`, then appends injected PII (card/phone/email/name/date/city/location) in STT form with filler context.
- **Mixed splits (`data_mix/*`)**: Synthetic hard-mode train + hybrid dev/test; `train_hardneg.jsonl` adds 200 unlabeled neg_XXXX hard negatives (PII-like spans left as O) to stress decoding filters.
- Regeneration commands and dataset notes live in `data_generation.md`.

## Modeling and Training
- Backbone: DistilBERT token classifier via `src/model.py` with `LABELS` from `src/labels.py`.
- Training script `src/train.py` (default 3 epochs, bs=8, lr=5e-5, max_len=256) with options for inverse-frequency class weights (`--class_weights`), focal vs CE loss (`--loss_type`), and optional grad clipping. Dataloading uses BIO tags from character spans (`src/dataset.py`).
- Trained artifacts:
  - `out_baseline`: Synthetic hard-mode, no decoding filters.
  - `out`: Synthetic hard-mode, tuned for precision; paired with decoding filters.
  - `out_weighted`: Synthetic hard-mode with class-weighted loss.
  - `out_hybrid`: Hybrid data (baseline and filtered decode variants share this model).
  - `out_mix`: Mixed synthetic-train + hybrid dev/test.
  - `out_mix_hardneg`: Mixed with hard negatives (collapsed on dev/test).

## Decoding and Precision Guards
- Inference `src/predict.py` supports global/per-label probability thresholds, card/phone minimum digit counts (spoken-number aware), optional EMAIL `at/dot` shape requirement, and PII flagging via `label_is_pii`. BIO spans decoded with attention/padding guards.
- Quantization helper `src/quantize_model.py` and latency probe `src/measure_latency.py` used to evaluate CPU p50/p95.

## Evaluation and Findings (see `metrics.md` for tables & artifact paths)
- **Synthetic hard-mode tuned (`out`, prob_threshold=0.8, filters on)**: Dev PII F1 0.951 (Macro-F1 0.966); p95 latency 14.50 ms (quantized p95 16.37 ms, no gain).
- **Synthetic baseline (`out_baseline`)**: Dev PII F1 0.966 without filters; slightly higher recall, lower precision resilience to near-misses.
- **Class-weighted (`out_weighted`)**: Dev PII F1 0.938, p95 14.69 ms; modest recall boost on minority labels, slightly lower precision than tuned.
- **Hybrid (`out_hybrid`)**: Near-perfect scores (Dev PII F1 ~1.00) with p95 16.66 ms; dynamic int8 quantization reduces p95 to 12.92 ms.
- **Mixed synthetic-train + hybrid dev/test (`out_mix`)**: Domain gap hurts PII precision (Dev PII F1 0.705 baseline, 0.649 with filters). Hard-negative variant (`out_mix_hardneg`) collapsed to all-O on dev/test despite fine train metrics.

## Artifacts and Repro Notes
- Prediction JSONs and latency outputs sit alongside each model directory (`out*/`), referenced in `metrics.md`.
- Data recipes and seeds are logged in `data_generation.md`; experiment-specific notes live in `improvements.md` and `metrics_*` files.
- Run `src/eval_span_f1.py` for span + PII metrics and `src/measure_latency.py` for latency verification on new checkpoints.
