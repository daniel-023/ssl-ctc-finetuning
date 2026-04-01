# Run Guide

## 1) HF dataset transcript field

```bash
ctc-build-vocab --config configs/train_hf_dataset_text.yaml
ctc-train --config configs/train_hf_dataset_text.yaml
```

## 2) HF audio + pseudolabel JSON

```bash
ctc-build-vocab --config configs/train_hf_audio_pseudolabel_json.yaml
ctc-train --config configs/train_hf_audio_pseudolabel_json.yaml
```

## 3) Local manifests

```bash
ctc-build-vocab --config configs/train_local_manifest_text.yaml
ctc-train --config configs/train_local_manifest_text.yaml
```

## 4) External ground-truth JSON join

```bash
ctc-build-vocab --config configs/train_hf_audio_external_gt_json.yaml
ctc-train --config configs/train_hf_audio_external_gt_json.yaml
```

## 5) Evaluation

```bash
ctc-eval --config configs/train_hf_dataset_text.yaml --set eval.model_dir=runs/xlsr300m_gt
ctc-eval --config configs/train_local_manifest_text.yaml --set eval.model_dir=runs/local_xlsr300m_gt
```

## 6) Plot Comparison

```bash
ctc-plot-compare --config configs/plot_compare.yaml
```

## Notes

- Every training run writes `resolved_config.yaml` and `train_command.sh` into `training.out_dir`.
- Use `--set key=value` for overrides.
- For local eval, manifest is auto-selected from `dataset.local.manifests.*` using `eval.split`.
