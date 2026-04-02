# Stage 03: Eval and Compare

## Purpose

Evaluate one trained run and optionally generate GT vs pseudolabel comparison plots.

## A) Evaluate one run

```bash
CONFIG=configs/train_hf_dataset_text.yaml
RUN_DIR=../runs/xlsr300m_gt

ctc-eval --config "$CONFIG" --set eval.model_dir="$RUN_DIR"
```

Dry-run check:

```bash
ctc-eval --config "$CONFIG" --dry-run
```

Expected outputs (paths set in config):
- `eval.out_json`
- `eval.out_jsonl` (if enabled)

## B) Compare GT vs pseudolabel runs

Set run paths in `configs/plot_compare.yaml`, then run:

```bash
ctc-plot-compare --config configs/plot_compare.yaml
```

Dry-run check:

```bash
ctc-plot-compare --config configs/plot_compare.yaml --dry-run
```

Expected outputs:
- `dev_wer_vs_global_step.png`
- `train_loss_vs_global_step.png`
- `final_test_wer_<variant>.png`
- `plot_summary.json`

## Common issues

- `No data available for plot`: missing eval points in `trainer_state.json`.
- `No final test WER values found`: missing `test_metrics.json` or expected keys in run folders.
