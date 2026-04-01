# Stage 03: Eval

## Goal
Evaluate a fine-tuned model and optionally compare two runs.

## 3.1 Evaluate one run
Command:
```bash
ctc-eval --config <CONFIG_YAML> --set eval.model_dir=<RUN_DIR>
```

Example:
```bash
ctc-eval --config configs/train_hf_dataset_text.yaml \
  --set eval.model_dir=runs/xlsr300m_gt
```

Outputs:
- `eval.out_json` (aggregate metrics)
- `eval.out_jsonl` (per-utterance predictions, if enabled)

## 3.2 Compare ground-truth vs pseudolabel runs
Command:
```bash
ctc-plot-compare --config configs/plot_compare.yaml
```

Outputs:
- `dev_wer_vs_global_step.png`
- `train_loss_vs_global_step.png`
- `final_test_wer_<variant>.png`
- `plot_summary.json`

## Quick checks
- Confirm WER metrics exist in eval output JSON.
- Confirm comparison plot files exist in `plot.out_dir`.
