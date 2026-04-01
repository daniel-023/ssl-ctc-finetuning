import argparse
import sys
from pathlib import Path

from ctc_framework.config.loader import apply_overrides, get_in, load_yaml
from ctc_framework.utils.subprocess import run_cmd


def parse_args():
    ap = argparse.ArgumentParser(description="Run evaluation from YAML config.")
    ap.add_argument("--config", required=True)
    ap.add_argument("--set", action="append", default=[])
    ap.add_argument("--dry-run", action="store_true")
    return ap.parse_args()


def main():
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[3]
    cfg = apply_overrides(load_yaml(args.config), args.set)

    script = repo_root / "legacy/ctc_finetuning/eval_infer_hf.py"
    if not script.exists():
        raise FileNotFoundError(f"Missing legacy eval script: {script}")

    model_dir = get_in(cfg, "eval.model_dir")
    if not model_dir:
        raise ValueError("eval.model_dir is required")

    cmd = [
        sys.executable,
        str(script),
        "--model_dir", str(model_dir),
        "--dataset", str(get_in(cfg, "dataset.name")),
        "--split", str(get_in(cfg, "eval.split")),
        "--audio_col", str(get_in(cfg, "eval.audio_col")),
        "--text_col", str(get_in(cfg, "eval.text_col")),
        "--id_col", str(get_in(cfg, "eval.id_col")),
        "--batch_size", str(get_in(cfg, "eval.batch_size")),
        "--num_workers", str(get_in(cfg, "eval.num_workers")),
        "--num_samples", str(get_in(cfg, "eval.num_samples")),
        "--max_sec", str(get_in(cfg, "eval.max_sec")),
        "--device", str(get_in(cfg, "eval.device", "auto")),
        "--out_json", str(get_in(cfg, "eval.out_json")),
    ]

    ds_config = get_in(cfg, "dataset.config")
    if ds_config is not None:
        cmd += ["--config", str(ds_config)]

    out_jsonl = get_in(cfg, "eval.out_jsonl")
    if out_jsonl:
        cmd += ["--out_jsonl", str(out_jsonl)]

    if get_in(cfg, "eval.discard_number_samples", False):
        cmd += ["--discard_number_samples"]

    if get_in(cfg, "eval.use_text_normalizer", True):
        cmd += ["--use_text_normalizer"]
    else:
        cmd += ["--no-use_text_normalizer"]

    normalizer_yaml = get_in(cfg, "eval.normalizer_yaml")
    if normalizer_yaml:
        cmd += ["--normalizer_yaml", str(normalizer_yaml)]

    if get_in(cfg, "eval.space_chinese_chars", True):
        cmd += ["--space_chinese_chars"]
    else:
        cmd += ["--no-space_chinese_chars"]

    if get_in(cfg, "eval.verbalize_numbers", False):
        cmd += ["--verbalize_numbers"]
    else:
        cmd += ["--no-verbalize_numbers"]

    run_cmd(cmd, cwd=repo_root, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
