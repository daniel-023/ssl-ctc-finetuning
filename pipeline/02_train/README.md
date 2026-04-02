# Stage 02: Train

## Purpose

Fine-tune XLSR-CTC using the vocab and transcript mode defined by config.

## Prerequisite

Run Stage 01 first for the same config so `vocab.out_path` exists.

## Command

```bash
CONFIG=configs/train_hf_dataset_text.yaml
ctc-train --config "$CONFIG"
```

Optional dry-run (checks config and data wiring without training):

```bash
ctc-train --config "$CONFIG" --dry-run
```

Override example:

```bash
ctc-train --config "$CONFIG" \
  --set training.out_dir=../runs/xlsr300m_gt_$(date +%Y%m%d_%H%M%S) \
  --set training.epochs=10
```

## Expected output

Inside `training.out_dir`:
- `resolved_config.yaml`
- `train_command.sh`
- `dataset_summary.json`
- checkpoints and final model files
- `test_metrics.json`

## Verify quickly

```bash
RUN_DIR=../runs/xlsr300m_gt
ls -la "$RUN_DIR"
cat "$RUN_DIR"/dataset_summary.json
cat "$RUN_DIR"/test_metrics.json
```

## Common issues

- `Vocab file not found`: run Stage 01 first, or fix `vocab.out_path`.
- OOM: reduce `training.bs` and/or increase `training.grad_accum`.
