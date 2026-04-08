# Stage 01: Data Prep (Build Vocab)

## Purpose

Build the character vocabulary file used by the CTC tokenizer. The vocabulary is collected by running all training transcripts through the text normalizer and extracting every unique character, then adding the special tokens `|` (word delimiter), `[UNK]`, and `[PAD]`.

## Input

- One config file from `configs/`
- Transcript source determined by config:
  - `transcript.source=inline` — reads from the dataset's text column
  - `transcript.source=jsonl, type=ground_truth` — reads from an external transcript JSONL
  - `vocab.mode=shared_hf_plus_pseudolabel` — also merges characters from a pseudolabel JSONL

## Command

```bash
CONFIG=configs/train_hf_dataset_text.yaml
ctc-build-vocab --config "$CONFIG"
```

Dry-run (validates paths and prints what would be done, without writing any files):

```bash
ctc-build-vocab --config "$CONFIG" --dry-run
```

## Expected output

Written to the directory of `vocab.out_path` (set in your config):
- `<vocab.out_path>` — character-to-index mapping used by the tokenizer (e.g. `vocab_shared.json`)
- `vocab_summary.json` — metadata: transcript source, normalizer settings, vocab size, output path

## Verify quickly

```bash
ls -la ../artifacts/vocab/
cat ../artifacts/vocab/vocab_summary.json
```

## Common issues

- **Path errors (`not found`)**: config paths are resolved relative to the config file location, not the working directory.
- **Empty or very small vocab**: the transcript text column may be wrong — check `dataset.columns.transcript` or `transcript.jsonl.text_col`.
- **Missing pseudolabel file**: if `vocab.mode=shared_hf_plus_pseudolabel`, set `transcript.jsonl.json_path` to a valid JSONL file.
