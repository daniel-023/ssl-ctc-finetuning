# Run Guide

The recommended run flow is stage-based:

1. `pipeline/01_data_prep/README.md`
2. `pipeline/02_train/README.md`
3. `pipeline/03_eval/README.md`

Use one of these configs throughout stages 01-03:

- `configs/train_hf_dataset_text.yaml`
- `configs/train_hf_audio_pseudolabel_json.yaml`
- `configs/train_hf_audio_external_gt_json.yaml`
- `configs/train_local_manifest_text.yaml`

Notes:

- Every training run writes `resolved_config.yaml` and `train_command.sh` into `training.out_dir`.
- Use `--set key=value` for overrides.
- For local eval, manifest is auto-selected from `dataset.local.manifests.*` using `eval.split`.
