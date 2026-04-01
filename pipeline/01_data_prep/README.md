# Stage 01: Data Prep

## Goal
Define the dataset + transcript source, then build the CTC vocabulary used for training.

## Inputs
- One training config under `configs/`
- Dataset backend in config:
  - `dataset.backend=hf` or `dataset.backend=local`
- Transcript source in config:
  - `transcript.source=inline`, or
  - `transcript.source=jsonl` with `transcript.jsonl.type=ground_truth|pseudolabel`

## Recommended configs
- Ground truth from HF dataset field:
  - `configs/train_hf_dataset_text.yaml`
- Pseudolabel JSON + HF audio:
  - `configs/train_hf_audio_pseudolabel_json.yaml`
- Ground truth JSON join:
  - `configs/train_hf_audio_external_gt_json.yaml`
- Local manifest mode:
  - `configs/train_local_manifest_text.yaml`

## Command
```bash
ctc-build-vocab --config <CONFIG_YAML>
```

Example:
```bash
ctc-build-vocab --config configs/train_hf_audio_pseudolabel_json.yaml
```

## What `ctc-build-vocab` does
- Loads transcript text according to config.
- Applies text normalization (if enabled).
- Collects unique characters.
- Adds CTC special tokens: `|`, `[UNK]`, `[PAD]`.
- Writes vocabulary JSON to `vocab.out_path`.

## Outputs
- Vocab JSON at `vocab.out_path`.
- `vocab_summary.json` in the same output directory.

## Quick checks
- Confirm vocab file exists at `vocab.out_path`.
- Confirm `vocab_summary.json` reports expected source and vocab size.
