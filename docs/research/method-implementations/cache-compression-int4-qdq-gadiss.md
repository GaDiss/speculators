---
title: "Compressed hidden-state caches: int4 backend and load-time QDQ for DFlash (@GaDiss)"
tags:
  - DFlash
  - Qwen3
  - hs_connectors
---

<div class="entry-byline" markdown>
[![GaDiss](https://github.com/GaDiss.png?size=64){ .entry-avatar }](https://github.com/GaDiss)
**[@GaDiss](https://github.com/GaDiss)** posted on **2026-11-23**
</div>

# Compressed hidden-state caches: int4 backend and load-time QDQ for DFlash (@GaDiss)

## At a glance

|                        |                                                                                                                                                                                                                             |
| ---------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Method**             | Group-wise int4 storage of extracted hidden states (`hs_connectors` backend `int4`), plus load-time fake quantization (`--cache-qdq-*`) with optional blockwise transforms after [WUSH](https://github.com/IST-DASLab/WUSH) |
| **Implemented in**     | Speculators + `hs_connectors`, branch `cache-compression` on top of `a01c3b85`                                                                                                                                              |
| **Landed as**          | not submitted                                                                                                                                                                                                               |
| **Scope**              | Offline training with the filesystem backends; any verifier whose hidden size is a multiple of the group size (128)                                                                                                         |
| **Not supported**      | Mooncake backend; int4 packing on the vLLM side fails loudly on non-finite hidden states                                                                                                                                    |
| **Correctness check**  | Consistency only: the int4 backend trains to the same loss as a bf16 cache with load-time int4 fake-quant; GPU-side and CPU-side packing are bit-identical. Draft quality under compression is not measured here            |
| **Status**             | prototype                                                                                                                                                                                                                   |
| **Smoke-test numbers** | int4 cache 4.4 GB vs bf16 17 GB for 200 Qwen3-8B samples                                                                                                                                                                    |

## Summary

Hidden-state caches can be stored at a quarter of the bf16 size (int4, group 128, fp16 scales) and training reads them unchanged; a load-time QDQ hook lets any scheme be tried on an existing bf16 cache without regeneration. Reached "runs". The checks below show the storage path and the fake-quant path agree with each other; they do not show that a draft trained on a compressed cache is as good as one trained on bf16. That needs a full-size training run and an acceptance-rate evaluation, which were not done here.

## Results

### Does it run?

Qwen3-8B verifier, DFlash draft (5 layers, block size 8, draft vocab 32k, target layers 2 18 33), 200 samples of `inference-optimization/speculators-ci-datasets:tutorial_regen` at 8192 tokens, one epoch on one L40S per run. Extraction with the `int4` and `file` backends, `generate-offline-data --validate-outputs` on both, five training runs, and the resulting checkpoint served by vLLM with 7 draft tokens.

### Is it correct?

Only consistency between the two code paths was checked. The training-loss rows come from 200 samples and one epoch, so they establish that the paths produce the same numbers, not that the method preserves draft quality.

| Check                                         | Reference                                                       | Result                  |
| --------------------------------------------- | --------------------------------------------------------------- | ----------------------- |
| int4 fake-quant vs physical pack/unpack       | unit test, 3 shapes                                             | bit-identical           |
| vLLM-side packing (GPU) vs `int4_pack` on CPU | same bf16 tensor                                                | bit-identical           |
| Training on the int4 cache                    | bf16 cache with `--cache-qdq-aux int4 --cache-qdq-teacher int4` | val loss 2.104 vs 2.105 |
| int4 aux + fp8 teacher, Hadamard transform    | bf16 baseline (2.107)                                           | runs, 2.108             |
| int4 aux + fp8 teacher, calibrated `wush025`  | bf16 baseline (2.107)                                           | runs, 2.107             |

Two extraction servers do not produce bit-identical raw hidden states, so an int4 cache and a bf16 cache from different servers differ at the code level (27% of elements, same 0.178 relative error from bf16 each); compare through the training loss or pack one server's bf16 output.

### Smoke-test performance

N/A.

## Observations

- Not yet measured in this repository: validation loss and acceptance rate of a full-size draft trained on an int4 cache versus a bf16 cache, and the effect of the teacher scheme. Until that is done, treat the int4 backend as a storage option whose quality cost is unknown and keep the teacher channel at fp8 or bf16 when in doubt.

## Setup

- Verifier: `Qwen/Qwen3-8B`, bf16 local copy
- Speculators: branch `cache-compression` (base `a01c3b85`); `hs_connectors` from the same branch on both sides
- vLLM 0.22.1 (needed `--no-enable-chunked-prefill`); training venv torch 2.14.1+cu130, transformers 5.19.0
- Hardware: NVIDIA L40S 46 GB, one per server or training run, driver 610.43

## Reproduction

`examples/train/dflash_qwen3_8b_offline_int4_cache.sh` with `MAX_SAMPLES=200`, `--on-missing skip` and `--max-anchors 1024`; set `HS_BACKEND=file` and `QDQ_ARGS` for the load-time variants, and `scripts/fit_cache_transform.py --data <output> --out <bundle>` before the `wush025` run. Training writes `run.yaml`, `train_command.txt` and `speculators.patch` next to the checkpoints; the launcher writes `vllm_command.txt` under `--provenance-dir`.
