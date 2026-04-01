import argparse
import shlex
import sys
from pathlib import Path

from ctc_framework.config.loader import apply_overrides, get_in, load_yaml
from ctc_framework.pipelines.common import resolve_path
from ctc_framework.pipelines.train_pipeline import run_training


def parse_args():
    ap = argparse.ArgumentParser(description="Run CTC training from YAML config.")
    ap.add_argument("--config", required=True)
    ap.add_argument("--set", action="append", default=[], help="Override config values: a.b.c=value")
    ap.add_argument("--dry-run", action="store_true")
    return ap.parse_args()


def main():
    args = parse_args()
    config_path = Path(args.config).resolve()
    cfg = apply_overrides(load_yaml(config_path), args.set)
    out_dir = resolve_path(get_in(cfg, "training.out_dir"), config_path)
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        invoked = ["ctc-train"] + sys.argv[1:]
        (out_dir / "train_command.sh").write_text(shlex.join(invoked) + "\n", encoding="utf-8")

    run_training(cfg, config_path, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
