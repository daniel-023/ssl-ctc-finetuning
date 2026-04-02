# ssl-ctc-finetuning

Config-driven SSL-CTC fine-tuning framework with XLSR-first defaults and reproducible experiment presets.

## What this repo covers

- Fine-tuning with Hugging Face or local datasets.
- Transcript source selection via config:
  - dataset transcript field (`transcript.source=inline`)
  - external JSONL join (`transcript.source=jsonl`)
- Ground-truth vs pseudolabel experiment support.
- Unified normalization support during vocab, train, and eval.
- Standardized outputs for run comparison and plotting.

## Installation

```bash
pip install -e .
```

## Recommended workflow

Run by stage (single source of truth for commands and checks):

1. Data prep: [`pipeline/01_data_prep/README.md`](pipeline/01_data_prep/README.md)
2. Train: [`pipeline/02_train/README.md`](pipeline/02_train/README.md)
3. Eval + compare plots: [`pipeline/03_eval/README.md`](pipeline/03_eval/README.md)

Shortcut overview: [`pipeline/README.md`](pipeline/README.md)  
Additional notes: [`docs/run_guide.md`](docs/run_guide.md), [`docs/data_schema.md`](docs/data_schema.md)

## Config presets

- `configs/train_hf_dataset_text.yaml`
  - HF audio + HF transcript field (ground truth)
  - Reproduces original setup with shared vocab augmentation from pseudolabel text.
- `configs/train_hf_audio_pseudolabel_json.yaml`
  - HF audio + pseudolabel JSONL transcripts joined by ID.
- `configs/train_hf_audio_external_gt_json.yaml`
  - HF audio + external ground-truth JSONL transcripts joined by ID.
- `configs/train_local_manifest_text.yaml`
  - Local JSONL manifests with inline transcript fields.
- `configs/plot_compare.yaml`
  - Ground-truth run vs pseudolabel run comparison plots.

Default pseudolabel file in this repo:
- `examples/pseudolabels/IMDA_pseudolabels.jsonl`

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
- `configs/`: experiment presets
- `pipeline/`: stage-based runbooks
- `examples/`: sample manifests/transcript inputs
- `docs/`: schema and usage notes
