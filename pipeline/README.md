# Pipeline

This project is run in 3 stages.

1. Build vocab from your selected transcript source.
2. Fine-tune XLSR-CTC.
3. Evaluate and optionally compare GT vs pseudolabel runs.

## Pick one training config first

- `configs/train_hf_dataset_text.yaml`: HF audio + HF text field (ground truth)
- `configs/train_hf_audio_pseudolabel_json.yaml`: HF audio + pseudolabel JSONL
- `configs/train_hf_audio_external_gt_json.yaml`: HF audio + external ground-truth JSONL
- `configs/train_local_manifest_text.yaml`: local JSONL manifests

## Run stages in order

- Stage 01: [`01_data_prep/README.md`](01_data_prep/README.md)
- Stage 02: [`02_train/README.md`](02_train/README.md)
- Stage 03: [`03_eval/README.md`](03_eval/README.md)

## Minimal end-to-end example

```bash
CONFIG=configs/train_hf_dataset_text.yaml

ctc-build-vocab --config "$CONFIG"
ctc-train --config "$CONFIG"
ctc-eval --config "$CONFIG" --set eval.model_dir=../runs/xlsr300m_gt
```

Use `configs/plot_compare.yaml` only when you already have both runs:
- GT run directory
- pseudolabel run directory
