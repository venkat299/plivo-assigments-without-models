# Improvements & Changes Log

- **Synthetic data hard-mode**: Regenerated `data/train.jsonl`, `data/dev.jsonl`, `data/test.jsonl` with `dev_mode=True` for all splits (seeds 42/123/321) to include held-out scenarios and near-miss noise.
- **Hybrid data**: Generated `data_hybrid/*` via `src/generate_hybrid_data.py` using real transcripts (`src/base_transcripts.txt`) + injected PII spans.
- **Predict decoding guards**: Added probability thresholding, digit/word count filters for card/phone, and email shape check in `src/predict.py`.
- **Class-weighted loss**: Added optional inverse-frequency class weights to `src/train.py` (`--class_weights`), trained weighted model `out_weighted`.
- **Model experiments**:
  - DistilBERT baseline/tuned on synthetic hard-mode data (`out_baseline`, `out`).
  - DistilBERT on hybrid data (baseline/tuned in `out_hybrid`).
  - DistilBERT with class weights on synthetic data (`out_weighted`).
  - Tiny BERT attempt (`prajjwal1/bert-tiny`) logged as underfitting (all-O).
- **Tracking docs**:
  - `metrics.md` for experiment tables (train/dev/test, latency).
  - `data_generation.md` documenting synthetic vs hybrid generation and CLI examples.
  - `Agents.md` updated to remind tracking dataset source/flags/seeds and updating data_generation.md.
