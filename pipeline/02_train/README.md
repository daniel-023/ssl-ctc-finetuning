# Stage 02: Train

## Goal
Fine-tune XLSR-CTC using the config and vocab prepared in Stage 01.

## Prerequisites
- Stage 01 completed successfully.
- `vocab.out_path` exists.

## Command
```bash
ctc-train --config <CONFIG_YAML>
```

Example:
```bash
ctc-train --config configs/train_hf_dataset_text.yaml
```

Override example:
```bash
ctc-train --config configs/train_hf_dataset_text.yaml \
  --set training.out_dir=runs/xlsr300m_gt_$(date +%Y%m%d_%H%M%S) \
  --set training.epochs=10
```

## Outputs
Inside `training.out_dir`:
- `resolved_config.yaml`
- `train_command.sh`
- `dataset_summary.json`
- Hugging Face trainer checkpoints
- Final model + processor files
- `test_metrics.json`

## Quick checks
- Confirm `dataset_summary.json` has expected train/dev/test sizes.
- Confirm `test_metrics.json` exists.
- Confirm best checkpoint metric matches `training.best_metric`.
