import argparse
import shlex
import sys
from pathlib import Path

from ctc_framework.config.loader import apply_overrides, get_in, load_yaml, save_yaml
from ctc_framework.utils.subprocess import run_cmd


def parse_args():
    ap = argparse.ArgumentParser(description="Run CTC training from YAML config.")
    ap.add_argument("--config", required=True)
    ap.add_argument("--set", action="append", default=[], help="Override config values: a.b.c=value")
    ap.add_argument("--dry-run", action="store_true")
    return ap.parse_args()


def main():
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[3]
    cfg = apply_overrides(load_yaml(args.config), args.set)

    train_script = repo_root / "legacy/ctc_finetuning/train_asr.py"
    if not train_script.exists():
        raise FileNotFoundError(f"Missing legacy train script: {train_script}")

    transcript_source = get_in(cfg, "transcripts.source", "dataset_field")
    if transcript_source == "dataset_field":
        train_source = "hf"
    elif transcript_source == "pseudo_json":
        train_source = "pseudo"
    else:
        raise ValueError("transcripts.source must be 'dataset_field' or 'pseudo_json'")

    out_dir = Path(get_in(cfg, "training.out_dir"))
    out_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        str(train_script),
        "--train_source", train_source,
        "--dataset", str(get_in(cfg, "dataset.name")),
        "--config", str(get_in(cfg, "dataset.config")) if get_in(cfg, "dataset.config") is not None else "",
        "--audio_col", str(get_in(cfg, "dataset.audio_col")),
        "--text_col", str(get_in(cfg, "dataset.text_col")),
        "--train_split", str(get_in(cfg, "dataset.train_split")),
        "--val_split", str(get_in(cfg, "dataset.val_split")),
        "--test_split", str(get_in(cfg, "dataset.test_split")),
        "--test_audio_col", str(get_in(cfg, "dataset.test_audio_col")),
        "--vocab", str(get_in(cfg, "vocab.out_path")),
        "--out", str(out_dir),
        "--max_sec", str(get_in(cfg, "training.max_sec")),
        "--num_proc", str(get_in(cfg, "training.num_proc")),
        "--seed", str(get_in(cfg, "training.seed")),
        "--val_size", str(get_in(cfg, "training.val_size")),
        "--bs", str(get_in(cfg, "training.bs")),
        "--grad_accum", str(get_in(cfg, "training.grad_accum")),
        "--lr", str(get_in(cfg, "training.lr")),
        "--epochs", str(get_in(cfg, "training.epochs")),
        "--lr_scheduler", str(get_in(cfg, "training.lr_scheduler")),
        "--warmup_ratio", str(get_in(cfg, "training.warmup_ratio")),
        "--weight_decay", str(get_in(cfg, "training.weight_decay")),
        "--max_grad_norm", str(get_in(cfg, "training.max_grad_norm")),
        "--eval_steps", str(get_in(cfg, "training.eval_steps")),
        "--save_steps", str(get_in(cfg, "training.save_steps")),
        "--logging_steps", str(get_in(cfg, "training.logging_steps")),
        "--save_total_limit", str(get_in(cfg, "training.save_total_limit")),
        "--best_metric", str(get_in(cfg, "training.best_metric")),
    ]

    test_text_col = get_in(cfg, "dataset.test_text_col")
    if test_text_col:
        cmd += ["--test_text_col", str(test_text_col)]

    if get_in(cfg, "normalization.use_text_normalizer", True):
        cmd += ["--use_text_normalizer"]
    else:
        cmd += ["--no-use_text_normalizer"]
    normalizer_yaml = get_in(cfg, "normalization.normalizer_yaml")
    if normalizer_yaml:
        cmd += ["--normalizer_yaml", str(normalizer_yaml)]

    if train_source == "pseudo":
        pseudo_jsonl = get_in(cfg, "transcripts.pseudo_jsonl")
        if not pseudo_jsonl:
            raise ValueError("transcripts.pseudo_jsonl is required for pseudo_json mode")
        cmd += [
            "--pseudo_jsonl", str(pseudo_jsonl),
            "--pseudo_audio_col", str(get_in(cfg, "transcripts.pseudo_audio_col")),
            "--pseudo_hf_id_col", str(get_in(cfg, "transcripts.pseudo_hf_id_col")),
            "--pseudo_hf_splits", str(get_in(cfg, "transcripts.pseudo_hf_splits")),
            "--pseudo_score_col", str(get_in(cfg, "transcripts.pseudo_score_col")),
            "--pseudo_min_score", str(get_in(cfg, "transcripts.pseudo_min_score")),
        ]
        pseudo_dev_jsonl = get_in(cfg, "transcripts.pseudo_dev_jsonl")
        if pseudo_dev_jsonl:
            cmd += ["--pseudo_dev_jsonl", str(pseudo_dev_jsonl)]

        if get_in(cfg, "transcripts.pseudo_join_on_hf_id", True):
            cmd += ["--pseudo_join_on_hf_id"]
        else:
            cmd += ["--no-pseudo_join_on_hf_id"]

        if get_in(cfg, "transcripts.disallow_test_split_in_pseudo_join", True):
            cmd += ["--disallow_test_split_in_pseudo_join"]
        else:
            cmd += ["--no-disallow_test_split_in_pseudo_join"]

    cmd = [x for x in cmd if x != ""]

    save_yaml(out_dir / "resolved_config.yaml", cfg)
    (out_dir / "train_command.sh").write_text(shlex.join(cmd) + "\n", encoding="utf-8")
    run_cmd(cmd, cwd=repo_root, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
