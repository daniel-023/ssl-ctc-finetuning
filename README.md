# ssl-ctc-finetuning

Config-driven SSL-CTC fine-tuning framework (XLSR-first defaults), with preserved legacy compatibility.

## Quick start

```bash
pip install -e .
```

Build vocab:

```bash
ctc-build-vocab --config configs/exp_gt_hf.yaml
```

Train (ground-truth HF transcript mode):

```bash
ctc-train --config configs/exp_gt_hf.yaml
```

Train (pseudolabel JSON + HF audio join mode):

```bash
ctc-build-vocab --config configs/exp_pseudo_json_hf_audio.yaml
ctc-train --config configs/exp_pseudo_json_hf_audio.yaml
```

### Overrides

```bash
ctc-train --config configs/exp_gt_hf.yaml \
  --set training.out_dir=runs/xlsr300m_gt_$(date +%Y%m%d_%H%M%S) \
  --set training.epochs=10
```

## Current status

- `src/ctc_framework/*` provides YAML-driven CLI wrappers.
- `legacy/*` preserves known working training/eval/plot scripts.
- Next phase: replace wrappers with native modular implementations while keeping config interface stable.
