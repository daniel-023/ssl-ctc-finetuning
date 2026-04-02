# Stage 01: Data Prep (Build Vocab)

## Purpose

Create the character vocabulary used by CTC training.

## Input

- One config file under `configs/`
- Correct transcript source in that config:
  - `transcript.source=inline` for dataset text field
  - `transcript.source=jsonl` for joined JSONL transcripts

## Command

```bash
CONFIG=configs/train_hf_dataset_text.yaml
ctc-build-vocab --config "$CONFIG"
```

Dry-run check:

```bash
ctc-build-vocab --config "$CONFIG" --dry-run
```

## Expected output

At `vocab.out_path` from config:
- vocab JSON file
- `vocab_summary.json`

## Verify quickly

```bash
ls -la ../artifacts/vocab
cat ../artifacts/vocab/vocab_summary.json
```

## Common issues

- `...not found` path errors: config paths are resolved relative to the config file.
- Empty or tiny vocab: transcript text column may be wrong (`dataset.columns.transcript` or `transcript.jsonl.text_col`).
