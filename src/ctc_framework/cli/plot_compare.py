import argparse
import sys
from pathlib import Path

from ctc_framework.config.loader import apply_overrides, get_in, load_yaml
from ctc_framework.utils.subprocess import run_cmd


def parse_args():
    ap = argparse.ArgumentParser(description="Plot run comparison from YAML config.")
    ap.add_argument("--config", required=True)
    ap.add_argument("--set", action="append", default=[])
    ap.add_argument("--dry-run", action="store_true")
    return ap.parse_args()


def main():
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[3]
    cfg = apply_overrides(load_yaml(args.config), args.set)

    script = repo_root / "legacy/ctc_finetuning/plot_compare_runs.py"
    if not script.exists():
        raise FileNotFoundError(f"Missing legacy plot script: {script}")

    cmd = [
        sys.executable,
        str(script),
        "--hf_run", str(get_in(cfg, "plot.hf_run")),
        "--pseudo_run", str(get_in(cfg, "plot.pseudo_run")),
        "--hf_label", str(get_in(cfg, "plot.hf_label", "Ground truth")),
        "--pseudo_label", str(get_in(cfg, "plot.pseudo_label", "Pseudolabel")),
        "--wer_variant", str(get_in(cfg, "plot.wer_variant", "norm")),
        "--out_dir", str(get_in(cfg, "plot.out_dir", "runs/compare_plots")),
    ]

    run_cmd(cmd, cwd=repo_root, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
