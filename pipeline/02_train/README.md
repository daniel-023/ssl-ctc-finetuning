# Stage 02: Train

## Purpose

Fine-tune XLSR-CTC using the vocab and transcript mode defined by config. Trains with the HuggingFace `Trainer` and saves the best checkpoint (lowest normalised WER on the dev set) plus final test metrics.

## Prerequisites

- Run Stage 01 first for the same config so `vocab.out_path` exists.
- A GPU with at least 16 GB VRAM is recommended for the default settings (`bs=16`, `grad_accum=2`). Reduce `training.bs` and increase `training.grad_accum` proportionally if you have less memory.

## Command

```bash
CONFIG=configs/train_hf_dataset_text.yaml
ctc-train --config "$CONFIG"
```

Dry-run (resolves all data sources and prints a dataset summary without starting training):

```bash
ctc-train --config "$CONFIG" --dry-run
```

Override example (timestamped run directory, shorter training):

```bash
ctc-train --config "$CONFIG" \
  --set training.out_dir=../runs/xlsr300m_gt_$(date +%Y%m%d_%H%M%S) \
  --set training.epochs=10
```

## Expected output

Inside `training.out_dir`:

| File | Description |
|------|-------------|
| `resolved_config.yaml` | Full merged config used for this run (defaults + overrides). Use to reproduce the run exactly. |
| `train_command.sh` | The exact CLI invocation that launched training. |
| `dataset_summary.json` | Split sizes, column mapping, normalizer settings. |
| `checkpoint-*/` | Intermediate checkpoints saved every `eval_steps`. |
| `pytorch_model.bin` / `model.safetensors` | Best model weights (loaded at end of training). |
| `test_metrics.json` | WER on the test split: `test_wer_raw_ref` (no normalisation) and `test_wer_norm_ref` (normalised). For the filler-stripped variant, run `ctc-eval` (Stage 03). |

## Verify quickly

```bash
RUN_DIR=../runs/xlsr300m_gt
cat "$RUN_DIR"/dataset_summary.json
cat "$RUN_DIR"/test_metrics.json
```

## Common issues

- **`Vocab file not found`**: run Stage 01 first, or fix `vocab.out_path` to match the generated file.
- **OOM (out of memory)**: reduce `training.bs` (e.g. to 4 or 8) and increase `training.grad_accum` to keep the effective batch size the same.
- **Very slow dataset loading**: text normalization always runs single-threaded (the normalizer is not fork-safe). Audio filtering uses `training.num_proc` workers in parallel.
