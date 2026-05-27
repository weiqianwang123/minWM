---
name: debug-world-model
description: "Diagnoses training/inference failures in video world model pipelines. Triggers on: training hang, loss NaN, shape mismatch, NCCL timeout, SP bug, distillation collapse, half-video noise. Use when user describes a symptom and wants root-cause diagnosis."
license: MIT
---

# debug-world-model

A debugging companion for the minWM project, drawing on hands-on experience across the full Causal Forcing / Causal Forcing++ pipeline. When you describe a symptom, this skill helps narrow down the likely root cause and points you toward a fix, with verification steps so you can confirm before changing code.

## Diagnosis Protocol

For each symptom, the suggested response is structured as:

1. **Pattern match** — which known failure mode this seems to correspond to
2. **Likely root cause** — a one-sentence explanation
3. **How to verify** — what to check before making changes
4. **Suggested fix** — what to consider changing, and where

---

## Known Failure Patterns

### Training Hang (stuck with no error output)

**Symptom**: All ranks wait indefinitely, no error logs.

**Root cause A — asymmetric rank behavior**: A rank skips a collective op due to a conditional branch; other ranks wait forever.
- Dangerous pattern: `if rank == 0: dist.xxx()` or `if grad_norm > threshold: pass` (evaluated independently per rank)
- Verify: search all `if.*rank.*==.*0` blocks for any `dist.` calls inside
- Fix: all collective calls (all-reduce, all-gather, broadcast) must be invoked on every rank, same count, same order

**Root cause B — wrong FSDP process group**: FSDP uses world group instead of DP group.
- Verify: check the `process_group` argument passed to FSDP initialization
- Fix: `FSDP(..., process_group=dp_process_group)`

**Root cause C — NCCL timeout during checkpoint save**: `dcp.save()` triggers heavy cross-node collective ops; slow shared storage IO causes timeout.
- Signature: fires at a fixed step, logs `WorkNCCL ... ran for 600000 milliseconds`, `NumelIn=1` means it's a scalar allreduce (barrier/grad_norm), not parameter sync
- Fix: remove `dcp.save()` entirely; keep only `gather_state_dict_on_cpu_rank0 + save_file`

---

### Loss Anomaly (NaN / no convergence / poor controllability)

**Symptom A — SP=1 works fine, SP≥2 loss immediately diverges**: RNG state fork.
- Root cause: one rank calls a random function one extra (or fewer) time inside the SP group; states diverge permanently
- Verify: search all `if rank == 0:` blocks for `torch.rand*` / `torch.randn*` / `dropout` calls
- Fix: all ranks must consume RNG, then broadcast rank 0's value
  ```python
  # Wrong
  if rank == 0:
      x = torch.randint(...)
  else:
      x = torch.empty(...)
  dist.broadcast(x, src=0)

  # Correct
  x = torch.randint(...)   # all ranks consume RNG
  dist.broadcast(x, src=0)
  ```

---

### Shape Mismatch / RuntimeError

**Symptom A — `expanded size (N) must match existing size (N//sp)` on KV cache write**:
- Root cause: after all-to-all, each rank has only `H // sp_size` heads, but the buffer was allocated with the full head count
- Rule: before allocating any buffer, ask — is this buffer used before or after all-to-all? After → divide head count by sp_size. Cross-attn does not go through SP all-to-all → head count unchanged.

**Symptom B — `block_mask was created for shape=(1,1,X,X) but got q_len=X+padding`**:
- Root cause: padding tokens added for sp_size divisibility enter attention, but block_mask was created for the original length
- Fix: strip padding before attention → run attention (block_mask matches stripped length) → restore padding after attention (zero-fill)
- Note: SP padding for video and text/action must follow the exact same logic, strictly aligned

---

### Inference Quality Issues

**Symptom A — generated video is half normal, half noise**:
- Root cause: `torch.randn` sampled independently per rank; sequence is corrupted after SP concatenation
- Fix: sample on rank 0, broadcast to all ranks

**Symptom B — EMA weights produce much worse results than training curves suggest**:
- Root cause: EMA save only stored a single rank's shard
- Fix: all-gather full parameters before saving EMA

---

### Miscellaneous

**adaln text timestep**: when caching text for AR inference, the text timestep must be fixed at 0 — it must not vary with the video timestep.

---

## Quick Triage Questions

If the symptom description is unclear, ask in priority order:

1. Is the symptom a hang / loss anomaly / shape error / inference quality issue?
2. Does SP=1 work correctly? (determines whether SP is involved)
3. Single-node or multi-node? Which stage triggers it (training / inference / checkpoint)?
4. Last few lines of the full traceback?
