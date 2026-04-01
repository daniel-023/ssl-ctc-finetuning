# ssl-ctc-finetuning

Config-driven SSL-CTC fine-tuning framework (XLSR-first defaults).

## Quick start

```bash
pip install -e .
```

## Pipeline stages

Run the project as 3 explicit stages:

1. Data prep: [`pipeline/01_data_prep/README.md`](pipeline/01_data_prep/README.md)
2. Train: [`pipeline/02_train/README.md`](pipeline/02_train/README.md)
3. Eval: [`pipeline/03_eval/README.md`](pipeline/03_eval/README.md)

Build vocab + train (HF dataset transcript field):

```bash
ctc-build-vocab --config configs/train_hf_dataset_text.yaml
ctc-train --config configs/train_hf_dataset_text.yaml
```

Note: this config reproduces your original experiment design by building a shared vocab
from HF transcript text + pseudolabel text (`vocab.mode=shared_hf_plus_pseudolabel`).

Build vocab + train (HF audio + pseudolabel JSON):

```bash
ctc-build-vocab --config configs/train_hf_audio_pseudolabel_json.yaml
ctc-train --config configs/train_hf_audio_pseudolabel_json.yaml
```

Default pseudolabel source in this repo:
- `examples/pseudolabels/IMDA_pseudolabels.jsonl`

Build vocab + train (local manifests with transcript field):

```bash
ctc-build-vocab --config configs/train_local_manifest_text.yaml
ctc-train --config configs/train_local_manifest_text.yaml
```

Build vocab + train (HF/local audio + external ground-truth JSON):

```bash
ctc-build-vocab --config configs/train_hf_audio_external_gt_json.yaml
ctc-train --config configs/train_hf_audio_external_gt_json.yaml
```

Evaluate:

```bash
ctc-eval --config configs/train_hf_dataset_text.yaml --set eval.model_dir=runs/xlsr300m_gt
ctc-eval --config configs/train_local_manifest_text.yaml --set eval.model_dir=runs/local_xlsr300m_gt
```

Plot comparison (ground-truth vs pseudolabel run):

```bash
ctc-plot-compare --config configs/plot_compare.yaml
```

### Override examples

```bash
ctc-train --config configs/train_hf_dataset_text.yaml \
  --set training.out_dir=runs/xlsr300m_gt_$(date +%Y%m%d_%H%M%S) \
  --set training.epochs=10
```

## Naming conventions

- `dataset.backend`: `hf` or `local`
- `dataset.hf_name` / `dataset.hf_config`
- `dataset.splits.*`
- `dataset.columns.*`
- `transcript.source`: `inline` or `jsonl`
- `transcript.jsonl.type`: `ground_truth` or `pseudolabel`
- `transcript.jsonl.*`
- `transcript.join.dataset_key` + `transcript.join.json_key` + `transcript.join.strict`
