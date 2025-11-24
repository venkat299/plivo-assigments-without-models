### Dev PII F1 — Summary

| Experiment                            | Dev PII F1 | Visual |
|---------------------------------------|-----------:|:-------|
| Synthetic Baseline                    |     0.966  | ██████████ |
| Synthetic Tuned (filters, thr=0.8)    |     0.951  | ██████████ |
| Hybrid Baseline                       |     1.000  | ██████████ |
| Hybrid Tuned (filters, thr=0.8)       |     1.000  | ██████████ |
| Synthetic Weighted (class weights)    |     0.938  | █████████▏ |
| Mixed Synthetic→Hybrid (baseline)     |     0.705  | ███████▎   |
| Mixed Synthetic→Hybrid (tuned)        |     0.649  | ██████▍    |
| Synthetic Focal + Grad Clip (collapsed)|    0.218  | ██▍        |


### Latency p50 (batch=1, CPU)

| Experiment / Model         | Latency p50 (ms) |
|----------------------------|-----------------:|
| Synthetic Baseline         |           12.38  |
| Synthetic Tuned            |           12.54  |
| Synthetic Weighted         |           12.73  |
| Synthetic Focal            |           —      |
| Hybrid Model (fp32)        |           14.08  |
| Hybrid Model (int8)        |            8.61  |
| Mixed Synthetic→Hybrid     |           14.63  |


```mermaid
bar
    title Dev PII F1 by experiment
    "Syn Baseline"         : 0.966
    "Syn Tuned"            : 0.951
    "Hybrid Baseline"      : 1.000
    "Hybrid Tuned"         : 1.000
    "Syn Weighted"         : 0.938
    "Mixed Base"           : 0.705
    "Mixed Tuned"          : 0.649
    "Focal + Clip"         : 0.218
```

### Dev PII F1 by model and dataset

| Model / Experiment                    | Synthetic (data/*) | Hybrid (data_hybrid/*) | Mixed Syn→Hybrid (data_mix/*) |
|--------------------------------------|-------------------:|-----------------------:|------------------------------:|
| Synthetic Baseline                   | 0.966              | —                      | —                             |
| Synthetic Tuned (filters, thr=0.8)   | 0.951              | —                      | —                             |
| Synthetic Weighted (class weights)   | 0.938              | —                      | —                             |
| Synthetic Focal + Grad Clip          | 0.218              | —                      | —                             |
| Hybrid Baseline                      | —                  | 1.000                  | —                             |
| Hybrid Tuned (filters, thr=0.8)      | —                  | 1.000                  | —                             |
| Mixed Baseline (filters off)         | —                  | —                      | 0.705                         |
| Mixed Tuned (filters on, thr=0.8)    | —                  | —                      | 0.649                         |
| Mixed Hard-neg (collapsed, tuned)    | —                  | —                      | 0.000                         |
