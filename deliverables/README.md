# Deliverables Bundle

- `dev_pred_synthetic_tuned.json`: Predictions on synthetic hard-mode dev (`data/dev.jsonl`) using `distilbert-base-uncased` with tuned decode (prob_threshold=0.8, filters on).
- `test_pred_synthetic_tuned.json`: Same model on synthetic hard-mode test (`data/test.jsonl`).
- `dev_pred_hybrid_tuned.json`: Predictions on hybrid dev (`data_hybrid/dev.jsonl`) with tuned decode.
- `test_pred_hybrid_tuned.json`: Same model on hybrid test (`data_hybrid/test.jsonl`).
- (Optional) `out/` and `out_hybrid/` contain full model+tokenizer checkpoints; `out_focal/` holds the focal-loss experiment (underfit).

Notes:
- Models live in `out/` (synthetic tuned) and `out_hybrid/` (hybrid tuned); see `metrics.md` for full metrics/latency.
- Decoding uses `prob_threshold=0.8`, min-digit/email shape filters as defined in `src/predict.py` defaults for tuned runs.
- Training options added: class weights, focal loss (`--loss_type focal`), gradient clipping (`--clip_grad_norm`), and per-label decode thresholds (`--label_thresholds`).

Current best models (per `metrics.md`):
- Hybrid tuned (`out_hybrid`): Dev/Test Macro-F1 ~0.999/1.0, PII F1 ~1.0, p95 ~16.7 ms (quantized p95 ~12.9 ms).
- Synthetic tuned (`out`): Dev Macro-F1 0.966, PII F1 0.951, p95 ~14.5 ms.
- Weighted CE (`out_weighted`): Dev PII F1 0.938; slightly below synthetic tuned.
- Mixed/focal/tiny/hardneg experiments are weaker; use only for reference.
