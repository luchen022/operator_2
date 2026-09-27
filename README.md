# XPU-OJ Operator 2 — MagiAttention / FFA

Private working repository for XPU-OJ 第一届算子优化比赛, problem 2.

## Current strategy

- Triton forward kernel for H800.
- Deterministic per-(Q block, Q head) ownership; no atomics in the timed path.
- Online softmax/LSE across active attention slices.
- Warmup-time preprocessing of static K/V, slice metadata, and Attention Sink state.
- Q-range partitioning so each timed kernel only evaluates slices that are active for that segment.

`main.py` is the current submission candidate. Scores and per-test timing feedback should be recorded here as iterations progress.
