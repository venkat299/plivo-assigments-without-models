# Model Metrics


## Baseline (distilbert-base-uncased, no filters, prob_threshold=0.5) — Hard-mode Train/Dev/Test (dev_mode=True)

| Split | Macro-F1 | PII Precision | PII Recall | PII F1 | Notes |
| --- | --- | --- | --- | --- | --- |
| Train (`data/train.jsonl` vs `out_baseline/train_pred.json`) | 0.982 | 0.976 | 0.978 | 0.977 | Hard-mode generator (dev noise) |
| Dev (`data/dev.jsonl` vs `out_baseline/dev_pred.json`) | 0.977 | 0.965 | 0.968 | 0.966 | Hard-mode generator |
| Test (`data/test.jsonl` vs `out_baseline/test_pred.json`) | 0.969 | 0.959 | 0.964 | 0.961 | Hard-mode generator |

Per-entity F1 (dev): CITY 1.000, CREDIT_CARD 0.973, DATE 0.960, EMAIL 0.976, LOCATION 1.000, PERSON_NAME 1.000, PHONE 0.926.

Latency (batch=1, CPU): p50 12.38 ms, p95 14.42 ms.

## Tuned (distilbert-base-uncased, filters on, prob_threshold=0.8)

| Split | Macro-F1 | PII Precision | PII Recall | PII F1 | Notes |
| --- | --- | --- | --- | --- | --- |
| Train (`data/train.jsonl` vs `out/train_pred.json`) | 0.969 | 0.969 | 0.960 | 0.964 | Hard-mode data; filters on |
| Dev (`data/dev.jsonl` vs `out/dev_pred.json`) | 0.966 | 0.953 | 0.950 | 0.951 | Hard-mode data; filters on |
| Test (`data/test.jsonl` vs `out/test_pred.json`) | 0.948 | 0.945 | 0.935 | 0.940 | Hard-mode data; filters on |

Per-entity F1 (dev): CITY 1.000, CREDIT_CARD 0.946, DATE 0.920, EMAIL 1.000, LOCATION 1.000, PERSON_NAME 1.000, PHONE 0.899.

Latency (batch=1, CPU): p50 12.54 ms, p95 14.50 ms.
Quantized latency (dynamic int8, CPU): p50 14.18 ms, p95 16.37 ms (no speedup vs fp32).

## Hybrid Baseline (distilbert-base-uncased, no filters, prob_threshold=0.5) — data_hybrid/*

| Split | Macro-F1 | PII Precision | PII Recall | PII F1 | Notes |
| --- | --- | --- | --- | --- | --- |
| Train (`data_hybrid/train.jsonl` vs `out_hybrid/train_pred_baseline.json`) | 1.000 | 1.000 | 1.000 | 1.000 | Hybrid injected into real transcripts |
| Dev (`data_hybrid/dev.jsonl` vs `out_hybrid/dev_pred_baseline.json`) | 0.998 | 1.000 | 1.000 | 1.000 | Slight non-PII drop (LOCATION) |
| Test (`data_hybrid/test.jsonl` vs `out_hybrid/test_pred_baseline.json`) | 1.000 | 1.000 | 1.000 | 1.000 |  |

Per-entity F1 (dev): CITY 1.000, CREDIT_CARD 1.000, DATE 1.000, EMAIL 1.000, LOCATION 0.983, PERSON_NAME 1.000, PHONE 1.000.

Latency (batch=1, CPU, hybrid model): p50 14.08 ms, p95 16.66 ms.
Quantized latency (dynamic int8, CPU): p50 8.61 ms, p95 12.92 ms.

## Hybrid Tuned (distilbert-base-uncased, filters on, prob_threshold=0.8) — data_hybrid/*

| Split | Macro-F1 | PII Precision | PII Recall | PII F1 | Notes |
| --- | --- | --- | --- | --- | --- |
| Train (`data_hybrid/train.jsonl` vs `out_hybrid/train_pred_tuned.json`) | 1.000 | 1.000 | 1.000 | 1.000 | Filters on |
| Dev (`data_hybrid/dev.jsonl` vs `out_hybrid/dev_pred_tuned.json`) | 0.999 | 1.000 | 1.000 | 1.000 | Filters on |
| Test (`data_hybrid/test.jsonl` vs `out_hybrid/test_pred_tuned.json`) | 0.998 | 1.000 | 1.000 | 1.000 | Filters on |

Per-entity F1 (dev): CITY 1.000, CREDIT_CARD 1.000, DATE 1.000, EMAIL 1.000, LOCATION 0.992, PERSON_NAME 1.000, PHONE 1.000.

Latency (batch=1, CPU, hybrid model): p50 14.08 ms, p95 16.66 ms (same model as baseline; decoding filters do not change model latency).

## Synthetic Weighted (distilbert-base-uncased, class_weights, filters on, prob_threshold=0.8) — data/*

| Split | Macro-F1 | PII Precision | PII Recall | PII F1 | Notes |
| --- | --- | --- | --- | --- | --- |
| Train (`out_weighted/train_pred.json`) | 0.962 | 0.941 | 0.956 | 0.949 | Class-weighted loss |
| Dev (`out_weighted/dev_pred.json`) | 0.958 | 0.929 | 0.948 | 0.938 | Class-weighted loss |
| Test (`out_weighted/test_pred.json`) | 0.951 | 0.931 | 0.945 | 0.938 | Class-weighted loss |

Per-entity F1 (dev): CITY 1.000, CREDIT_CARD 0.959, DATE 0.866, EMAIL 0.984, LOCATION 0.984, PERSON_NAME 0.991, PHONE 0.923.

Latency (batch=1, CPU, weighted model): p50 12.73 ms, p95 14.69 ms.

## Synthetic Focal + Grad Clip (distilbert-base-uncased, class_weights, focal gamma=2.0, clip=1.0, prob_threshold=0.8, filters on) — data/*

Model underfit/collapsed on PII (mostly O except CITY/LOCATION/NAME).

| Split | Macro-F1 | PII Precision | PII Recall | PII F1 | Notes |
| --- | --- | --- | --- | --- | --- |
| Train (`out_focal/train_pred.json`) | 0.362 | 0.215 | 0.224 | 0.219 | PII mostly dropped |
| Dev (`out_focal/dev_pred.json`) | 0.355 | 0.213 | 0.224 | 0.218 | PII mostly dropped |
| Test (`out_focal/test_pred.json`) | 0.368 | 0.226 | 0.213 | 0.219 | PII mostly dropped |

Lowering threshold to 0.5 improved recall but remained weak (dev Macro-F1 0.464, PII F1 0.416).

## Mixed Synthetic Train + Hybrid Dev/Test (distilbert-base-uncased) — data_mix/*

### Baseline decode (prob_threshold=0.5, filters off via min-digits=0)

| Split | Macro-F1 | PII Precision | PII Recall | PII F1 | Notes |
| --- | --- | --- | --- | --- | --- |
| Train (`out_mix/train_pred_base.json`) | 0.978 | 0.970 | 0.972 | 0.971 | Synthetic train, hybrid dev/test |
| Dev (`out_mix/dev_pred_base.json`) | 0.741 | 0.666 | 0.750 | 0.705 | Weak on card/phone precision |
| Test (`out_mix/test_pred_base.json`) | 0.732 | 0.646 | 0.753 | 0.695 | Weak on card/phone precision |

Per-entity F1 (dev): CITY 1.000, CREDIT_CARD 0.254, DATE 1.000, EMAIL 0.756, LOCATION 0.909, PERSON_NAME 0.963, PHONE 0.305.

### Tuned decode (prob_threshold=0.8, filters on: email shape + min digits)

| Split | Macro-F1 | PII Precision | PII Recall | PII F1 | Notes |
| --- | --- | --- | --- | --- | --- |
| Train (`out_mix/train_pred_tuned.json`) | 0.950 | 0.952 | 0.950 | 0.951 | Filters on |
| Dev (`out_mix/dev_pred_tuned.json`) | 0.679 | 0.732 | 0.582 | 0.649 | Filters hurt recall; precision mixed |
| Test (`out_mix/test_pred_tuned.json`) | 0.675 | 0.739 | 0.588 | 0.655 | Filters hurt recall; precision mixed |

Per-entity F1 (dev): CITY 0.935, CREDIT_CARD 0.262, DATE 1.000, EMAIL 0.376, LOCATION 0.714, PERSON_NAME 0.972, PHONE 0.492.

Latency (batch=1, CPU, out_mix model): p50 14.63 ms, p95 16.39 ms.

### Hard-negative train (synthetic train + 200 hybrid O-only examples) — model `out_mix_hardneg`

Tuned decode (prob_threshold=0.8/0.5, filters on) collapsed to all-O on dev/test; not a viable improvement.

| Split | Macro-F1 | PII Precision | PII Recall | PII F1 | Notes |
| --- | --- | --- | --- | --- | --- |
| Train (`out_mix_hardneg/train_pred_tuned.json`) | 0.962 | 0.965 | 0.962 | 0.964 | Train still fine |
| Dev (`out_mix_hardneg/dev_pred_tuned.json`) | 0.000 | 0.000 | 0.000 | 0.000 | All-O |
| Test (`out_mix_hardneg/test_pred_tuned.json`) | 0.000 | 0.000 | 0.000 | 0.000 | All-O |

Lowering threshold to 0.5 did not recover dev/test (still all-O).
