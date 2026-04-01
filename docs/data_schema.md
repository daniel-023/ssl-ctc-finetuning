# Data Schemas

## Local Manifest Schema

Each line in train/val/test manifest JSONL should include:

- `id` (string): utterance identifier used for joins/debugging.
- `<dataset.columns.audio>` (string or Audio object): path to WAV/FLAC or decoded HF-style audio object.
- `<dataset.columns.transcript>` (string): transcript text when `transcript.source=dataset_field`.

Example (`dataset.columns.audio=audio_path`, `dataset.columns.transcript=text`):

```json
{"id":"utt-0001","audio_path":"/abs/path/utt-0001.wav","text":"hello world"}
```

If paths are relative, set `dataset.local.audio_root`.

## External Transcript JSON Schema

When `transcript.source=external_gt_json`, provide one JSONL with:

- `<transcript.external_gt.id_col>`: transcript join key.
- `<transcript.external_gt.text_col>`: transcript text.

Example:

```json
{"id":"utt-0001","text":"hello world"}
```

Config controls join behavior:

- `transcript.join.dataset_key`
- `transcript.external_gt.id_col`
- `transcript.external_gt.text_col`
- `transcript.join.strict`

If `transcript.join.strict=true`, unmatched IDs raise an error.
