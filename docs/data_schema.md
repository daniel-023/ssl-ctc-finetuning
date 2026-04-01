# Data Schemas

## Local Manifest Schema

Each line in train/val/test manifest JSONL should include:

- `id` (string): utterance identifier used for joins/debugging.
- `<dataset.columns.audio>` (string or Audio object): path to WAV/FLAC or decoded HF-style audio object.
- `<dataset.columns.transcript>` (string): transcript text when `transcript.source=inline`.

Example (`dataset.columns.audio=audio_path`, `dataset.columns.transcript=text`):

```json
{"id":"utt-0001","audio_path":"/abs/path/utt-0001.wav","text":"hello world"}
```

If paths are relative, set `dataset.local.audio_root`.

## Ground-Truth JSON Schema

When `transcript.source=jsonl` and `transcript.jsonl.type=ground_truth`, provide one JSONL with:

- `<transcript.join.json_key>`: transcript join key.
- `<transcript.jsonl.text_col>`: transcript text.

Example:

```json
{"id":"utt-0001","text":"hello world"}
```

Config controls join behavior:

- `transcript.join.dataset_key`
- `transcript.join.json_key`
- `transcript.jsonl.text_col`
- `transcript.join.strict`

If `transcript.join.strict=true`, unmatched IDs raise an error.

## Pseudolabel JSON Schema

When `transcript.source=jsonl` and `transcript.jsonl.type=pseudolabel`, provide one JSONL with:

- `<transcript.jsonl.text_col>`: pseudolabel text (default `text`)
- `<transcript.jsonl.audio_path_col>`: audio path or filename used to derive utterance ID (default `audio_path`)
- `<transcript.jsonl.score_col>`: optional confidence score used for filtering (default `score`)

Example:

```json
{"audio_path":"imda-2021-part6-00001-channel001m-0000000-0000430.wav","text":"hello world","score":0.93}
```

Default pseudolabel file included in this repo:
- `examples/pseudolabels/IMDA_pseudolabels.jsonl`
