import argparse
import sys
from pathlib import Path

from ctc_framework.config.loader import apply_overrides, get_in, load_yaml
from ctc_framework.utils.subprocess import run_cmd


def parse_args():
    ap = argparse.ArgumentParser(description="Build CTC vocab from YAML config.")
    ap.add_argument("--config", required=True)
    ap.add_argument("--set", action="append", default=[])
    ap.add_argument("--dry-run", action="store_true")
    return ap.parse_args()


def main():
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[3]
    cfg = apply_overrides(load_yaml(args.config), args.set)

    script = repo_root / "legacy/ctc_finetuning/make_vocab.py"
    if not script.exists():
        raise FileNotFoundError(f"Missing legacy vocab script: {script}")

    mode = get_in(cfg, "vocab.mode", "dataset_only")
    dataset = str(get_in(cfg, "dataset.name"))
    ds_config = get_in(cfg, "dataset.config")
    train_split = str(get_in(cfg, "dataset.train_split"))
    text_col = str(get_in(cfg, "dataset.text_col"))

    cmd = [
        sys.executable,
        str(script),
        "--dataset", dataset,
        "--train_split", train_split,
        "--text_col", text_col,
        "--score_col", str(get_in(cfg, "vocab.score_col", "score")),
        "--min_score", str(get_in(cfg, "vocab.min_score", 0.0)),
        "--out", str(get_in(cfg, "vocab.out_path")),
    ]
    if ds_config is not None:
        cmd += ["--config", str(ds_config)]

    if mode == "shared_hf_plus_pseudo":
        pseudo_jsonl = get_in(cfg, "transcripts.pseudo_jsonl")
        if not pseudo_jsonl:
            raise ValueError("transcripts.pseudo_jsonl is required for vocab.mode=shared_hf_plus_pseudo")
        cmd += [
            "--aux_jsonl", str(pseudo_jsonl),
            "--aux_text_col", text_col,
            "--aux_score_col", str(get_in(cfg, "vocab.aux_score_col", "score")),
            "--aux_min_score", str(get_in(cfg, "vocab.aux_min_score", get_in(cfg, "transcripts.pseudo_min_score", 0.0))),
        ]

    if get_in(cfg, "normalization.use_text_normalizer", True):
        cmd += ["--use_text_normalizer"]
    else:
        cmd += ["--no-use_text_normalizer"]
    normalizer_yaml = get_in(cfg, "normalization.normalizer_yaml")
    if normalizer_yaml:
        cmd += ["--normalizer_yaml", str(normalizer_yaml)]

    run_cmd(cmd, cwd=repo_root, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
