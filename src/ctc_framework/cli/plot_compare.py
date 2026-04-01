import argparse
from pathlib import Path

from ctc_framework.config.loader import apply_overrides, load_yaml
from ctc_framework.pipelines.plot_pipeline import run_plot_compare


def parse_args():
    ap = argparse.ArgumentParser(description="Plot run comparison from YAML config.")
    ap.add_argument("--config", required=True)
    ap.add_argument("--set", action="append", default=[])
    ap.add_argument("--dry-run", action="store_true")
    return ap.parse_args()


def main():
    args = parse_args()
    config_path = Path(args.config).resolve()
    cfg = apply_overrides(load_yaml(config_path), args.set)
    run_plot_compare(cfg, config_path, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
