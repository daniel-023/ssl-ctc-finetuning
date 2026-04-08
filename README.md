# ssl-ctc-finetuning

Config-driven framework for fine-tuning wav2vec 2.0 / XLSR models for ASR using CTC, with reproducible experiment presets. Based on Baevski et al. (2020).

## What this repo covers

- Fine-tuning with Hugging Face or local datasets.
- Transcript source selection via config:
  - dataset transcript field (`transcript.source=inline`)
  - external JSONL join (`transcript.source=jsonl`)
- Ground-truth vs pseudolabel experiment support.
- Unified normalization support during vocab, train, and eval.
- Standardized outputs for run comparison and plotting.

## Experimental setup

### Model
wav2vec 2.0 large (XLSR-300M) with a single linear CTC projection head. Pre-trained weights: `facebook/wav2vec2-xls-r-300m` (Baevski et al., 2020; Conneau et al., 2020).

### Training hyperparameters

| Setting | Value |
|---------|-------|
| Optimizer | AdamW |
| Learning rate | 3e-5 |
| LR scheduler | Cosine with 10% linear warmup |
| Effective batch size | 32 (16 per device × 2 gradient accumulation steps) |
| Epochs | 20 |
| Weight decay | 0.01 |
| Max gradient norm | 1.0 |
| Best checkpoint | Lowest normalised WER on dev set |
| Max audio duration | 30 s (longer clips discarded) |

### Text normalization
Transcripts are lowercased, punctuation-stripped, and optionally filler-word-filtered (e.g. "uh", "um") using a configurable filler list. The same normalizer runs identically across vocab build, training, and evaluation.

### Evaluation metrics
Three WER variants are reported:
- `wer_raw_ref` — raw prediction vs raw reference (no normalisation)
- `wer_norm_ref` — normalised prediction vs normalised reference
- `wer_norm_no_fill_ref` — normalised prediction vs filler-stripped reference (strictest)

## Installation

```bash
pip install -e .
```

## Recommended workflow

Run each stage in order. Use `--dry-run` first to validate paths and data before committing.

1. Build vocab: [`pipeline/01_data_prep/README.md`](pipeline/01_data_prep/README.md)
2. Train: [`pipeline/02_train/README.md`](pipeline/02_train/README.md)
3. Eval + compare: [`pipeline/03_eval/README.md`](pipeline/03_eval/README.md)

Quick overview: [`pipeline/README.md`](pipeline/README.md) — data format reference: [`docs/data_schema.md`](docs/data_schema.md)

## Config presets

| Config | Audio source | Transcript source |
|--------|-------------|-------------------|
| `train_hf_dataset_text.yaml` | HuggingFace dataset | Inline text column (ground truth) |
| `train_hf_audio_pseudolabel_json.yaml` | HuggingFace dataset | Pseudolabel JSONL joined by ID |
| `train_hf_audio_external_gt_json.yaml` | HuggingFace dataset | External ground-truth JSONL joined by ID |
| `train_local_manifest_text.yaml` | Local JSONL manifest files | Inline text column |
| `plot_compare.yaml` | — | GT vs pseudolabel run comparison plots |

Default pseudolabel file: `examples/pseudolabels/IMDA_pseudolabels.jsonl`

## Reproducibility conventions

- Every training run writes:
  - `resolved_config.yaml`
  - `train_command.sh`
  - `dataset_summary.json`
  - checkpoints and `test_metrics.json`
- Use CLI overrides for controlled variations:
  - `--set key=value`

## Repository map

- `src/ctc_framework/`: framework code and CLIs
- `configs/`: experiment config presets
- `pipeline/`: stage-by-stage runbooks
- `tests/`: pytest test suite (`pip install -e ".[dev]"`, then `pytest`)
- `examples/`: sample manifests and pseudolabel inputs
- `docs/`: data schema reference
