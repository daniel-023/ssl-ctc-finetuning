# Stage 03: Eval and Compare

## Purpose

Evaluate one trained model and optionally generate comparison plots between a ground-truth run and a pseudolabel run.

## A) Evaluate one run

```bash
CONFIG=configs/train_hf_dataset_text.yaml
RUN_DIR=../runs/xlsr300m_gt

ctc-eval --config "$CONFIG" --set eval.model_dir="$RUN_DIR"
```

Dry-run (resolves dataset and normalizer config without running inference):

```bash
ctc-eval --config "$CONFIG" --dry-run
```

### Output files

| File | Set by | Description |
|------|--------|-------------|
| `eval_outputs/metrics.json` | `eval.out_json` | WER metrics (see below) |
| `eval_outputs/preds.jsonl` | `eval.out_jsonl` | Per-utterance predictions and references (written only when `eval.out_jsonl` is set in config) |

### Metrics in `metrics.json`

| Key | Description |
|-----|-------------|
| `wer_raw_ref` | WER: raw predictions vs raw references (no normalisation) |
| `wer_norm_ref` | WER: normalised predictions vs normalised references |
| `wer_norm_no_fill_ref` | WER: normalised predictions vs references with fillers removed (strictest) |

## B) Compare GT vs pseudolabel runs

Edit `configs/plot_compare.yaml` to set both run directories, then run:

```bash
ctc-plot-compare --config configs/plot_compare.yaml
```

Dry-run check:

```bash
ctc-plot-compare --config configs/plot_compare.yaml --dry-run
```

Both run directories must contain `trainer_state.json` (written by HuggingFace Trainer during training) and `test_metrics.json`.

### Output files

| File | Description |
|------|-------------|
| `dev_wer_vs_global_step.png` | Dev WER curves over training steps for both runs |
| `train_loss_vs_global_step.png` | Training loss curves |
| `final_test_wer_<variant>.png` | Bar chart of final test WER for both runs |
| `plot_summary.json` | Extracted metrics used for the plots |

## Common issues

- **`No data available for plot`**: `trainer_state.json` has no eval entries — check that `eval_steps` was set during training and training ran long enough to reach the first eval checkpoint.
- **`No final test WER values found`**: `test_metrics.json` is missing or doesn't contain the expected keys — verify training completed successfully.
