import ast
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from ctc_framework.config.loader import get_in
from ctc_framework.pipelines.common import resolve_path


def _safe_float(value) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def find_trainer_state(run_dir: Path) -> Optional[Path]:
    direct = run_dir / "trainer_state.json"
    if direct.exists():
        return direct

    ckpt_states = list(run_dir.glob("checkpoint-*/trainer_state.json"))
    if not ckpt_states:
        return None

    def _ckpt_step(path: Path) -> int:
        match = re.search(r"checkpoint-(\d+)$", path.parent.name)
        return int(match.group(1)) if match else -1

    ckpt_states.sort(key=_ckpt_step)
    return ckpt_states[-1]


def load_log_history(run_dir: Path) -> List[Dict]:
    state_path = find_trainer_state(run_dir)
    if state_path is None:
        return []

    payload = json.loads(state_path.read_text(encoding="utf-8"))
    history = payload.get("log_history", [])
    return history if isinstance(history, list) else []


def extract_series(log_history: List[Dict], metric_key: str) -> List[Tuple[int, float]]:
    by_step: Dict[int, float] = {}
    for row in log_history:
        if metric_key not in row or "step" not in row:
            continue
        step = _safe_float(row.get("step"))
        value = _safe_float(row.get(metric_key))
        if step is None or value is None:
            continue
        by_step[int(step)] = value
    return sorted(by_step.items(), key=lambda x: x[0])


def read_test_wer_from_file(run_dir: Path, preferred_keys: List[str]) -> Optional[Tuple[float, str]]:
    metrics_path = run_dir / "test_metrics.json"
    if not metrics_path.exists():
        return None

    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    for key in preferred_keys:
        if key in payload:
            value = _safe_float(payload[key])
            if value is not None:
                return value, f"{metrics_path.name}:{key}"
    return None


def read_test_wer_from_history(log_history: List[Dict], preferred_keys: List[str]) -> Optional[Tuple[float, str]]:
    for row in reversed(log_history):
        for key in preferred_keys:
            if key in row:
                value = _safe_float(row.get(key))
                if value is not None:
                    return value, f"trainer_state:{key}"
    return None


def read_test_wer_from_training_log(run_dir: Path, preferred_keys: List[str]) -> Optional[Tuple[float, str]]:
    log_path = run_dir / "training.log"
    if not log_path.exists():
        return None

    last_metrics_line = None
    with log_path.open("r", encoding="utf-8", errors="ignore") as fin:
        for line in fin:
            if "TEST metrics:" in line:
                last_metrics_line = line.strip()

    if not last_metrics_line:
        return None

    match = re.search(r"TEST metrics:\s*(\{.*\})", last_metrics_line)
    if not match:
        return None

    try:
        payload = ast.literal_eval(match.group(1))
    except (SyntaxError, ValueError):
        return None

    if not isinstance(payload, dict):
        return None

    for key in preferred_keys:
        if key in payload:
            value = _safe_float(payload[key])
            if value is not None:
                return value, f"training.log:{key}"
    return None


def read_test_wer_with_fallback(
    run_dir: Path,
    log_history: List[Dict],
    preferred_keys: List[str],
    eval_curve_key: str,
) -> Tuple[Optional[float], str]:
    from_file = read_test_wer_from_file(run_dir, preferred_keys)
    if from_file:
        return from_file

    from_history = read_test_wer_from_history(log_history, preferred_keys)
    if from_history:
        return from_history

    from_log = read_test_wer_from_training_log(run_dir, preferred_keys)
    if from_log:
        return from_log

    eval_curve = extract_series(log_history, eval_curve_key)
    if eval_curve:
        return eval_curve[-1][1], f"fallback:last_{eval_curve_key}"

    return None, "missing"


def load_run(run_dir: Path, wer_variant: str) -> Dict:
    history = load_log_history(run_dir)
    if wer_variant == "norm":
        dev_keys = ["eval_wer_decoded_norm", "eval_wer"]
        test_keys = ["test_wer_norm_ref", "test_wer_decoded_norm", "test_wer", "eval_wer", "wer"]
    else:
        dev_keys = ["eval_wer_decoded_label_ref", "eval_wer_decoded_raw", "eval_wer"]
        test_keys = [
            "test_wer_raw_ref",
            "test_wer_decoded_label_ref",
            "test_wer_decoded_raw",
            "test_wer",
            "eval_wer",
            "wer",
        ]

    dev_wer = []
    dev_key_used = "missing"
    for key in dev_keys:
        dev_wer = extract_series(history, key)
        if dev_wer:
            dev_key_used = key
            break

    train_loss = extract_series(history, "loss")
    test_wer, source = read_test_wer_with_fallback(
        run_dir,
        history,
        test_keys,
        dev_key_used if dev_key_used != "missing" else "eval_wer",
    )

    return {
        "history_len": len(history),
        "dev_wer_key": dev_key_used,
        "dev_wer": dev_wer,
        "train_loss": train_loss,
        "test_wer": test_wer,
        "test_wer_source": source,
    }


def plot_line(plt, out_path: Path, title: str, ylabel: str, gt_label: str, gt_data, pseudo_label: str, pseudo_data):
    fig, ax = plt.subplots(figsize=(8, 5))
    have_data = False
    y_values: List[float] = []
    endpoints: List[Tuple[float, float, str]] = []

    if gt_data:
        x = [p[0] for p in gt_data]
        y = [p[1] for p in gt_data]
        ax.plot(x, y, label=gt_label, linewidth=2.2, marker="o", markersize=3.5, alpha=0.95)
        ax.scatter([x[-1]], [y[-1]], s=28, zorder=4)
        endpoints.append((x[-1], y[-1], f"{y[-1]:.4f}"))
        y_values.extend(y)
        have_data = True

    if pseudo_data:
        x = [p[0] for p in pseudo_data]
        y = [p[1] for p in pseudo_data]
        ax.plot(x, y, label=pseudo_label, linewidth=2.2, marker="s", markersize=3.5, alpha=0.95)
        ax.scatter([x[-1]], [y[-1]], s=28, zorder=4)
        endpoints.append((x[-1], y[-1], f"{y[-1]:.4f}"))
        y_values.extend(y)
        have_data = True

    if not have_data:
        raise ValueError(f"No data available for plot: {title}")

    ax.set_title(title)
    ax.set_xlabel("Global Step")
    ax.set_ylabel(ylabel)
    if ylabel == "WER" and y_values:
        y_min = min(y_values)
        y_max = max(y_values)
        span = y_max - y_min
        pad = max(span * 0.15, 0.002)
        ax.set_ylim(max(0.0, y_min - pad), min(1.0, y_max + pad))

    offsets = [(10, 8)] * len(endpoints)
    if len(endpoints) == 2:
        close = abs(endpoints[0][1] - endpoints[1][1]) < 0.01
        if close:
            high_idx = 0 if endpoints[0][1] >= endpoints[1][1] else 1
            low_idx = 1 - high_idx
            offsets = [(10, 8), (10, 8)]
            offsets[high_idx] = (10, 12)
            offsets[low_idx] = (10, -22)

    for (x_end, y_end, text), (dx, dy) in zip(endpoints, offsets):
        ax.annotate(
            text,
            (x_end, y_end),
            textcoords="offset points",
            xytext=(dx, dy),
            fontsize=8,
            zorder=6,
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.9),
            arrowprops=dict(arrowstyle="-", color="gray", lw=0.8, alpha=0.7),
        )

    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_test_bar(plt, out_path: Path, gt_label: str, gt_wer: Optional[float], pseudo_label: str, pseudo_wer: Optional[float]):
    labels = []
    values = []
    colors = []

    if gt_wer is not None:
        labels.append(gt_label)
        values.append(gt_wer)
        colors.append("#1f77b4")
    if pseudo_wer is not None:
        labels.append(pseudo_label)
        values.append(pseudo_wer)
        colors.append("#ff7f0e")

    if not values:
        raise ValueError("No final test WER values found for either run.")

    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(labels, values, color=colors)
    ax.set_title("Final Test WER (Lower Is Better)")
    ax.set_ylabel("WER")
    ax.grid(axis="y", alpha=0.3)

    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height(),
            f"{val:.4f}",
            ha="center",
            va="bottom",
        )

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_dev_gap(plt, out_path: Path, gt_data, pseudo_data, gt_label: str, pseudo_label: str):
    gt_map = {int(step): float(val) for step, val in gt_data}
    pseudo_map = {int(step): float(val) for step, val in pseudo_data}
    common_steps = sorted(set(gt_map.keys()) & set(pseudo_map.keys()))
    if not common_steps:
        raise ValueError("No overlapping steps between ground-truth and pseudolabel dev curves.")

    gap = [pseudo_map[s] - gt_map[s] for s in common_steps]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(common_steps, gap, linewidth=2.2, color="#2ca02c", marker="d", markersize=3.5)
    ax.axhline(0.0, color="black", linewidth=1.0, linestyle="--", alpha=0.7)
    ax.set_title(f"Dev WER Gap: {pseudo_label} - {gt_label}")
    ax.set_xlabel("Global Step")
    ax.set_ylabel("WER Gap")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def run_plot_compare(cfg: dict, config_path: Path, dry_run: bool = False):
    gt_run = resolve_path(get_in(cfg, "plot.ground_truth_run"), config_path)
    pseudo_run = resolve_path(get_in(cfg, "plot.pseudolabel_run"), config_path)
    if gt_run is None or pseudo_run is None:
        raise ValueError("plot.ground_truth_run and plot.pseudolabel_run are required")
    if not gt_run.exists():
        raise FileNotFoundError(f"Ground-truth run directory not found: {gt_run}")
    if not pseudo_run.exists():
        raise FileNotFoundError(f"Pseudolabel run directory not found: {pseudo_run}")

    gt_label = str(get_in(cfg, "plot.ground_truth_label", "Ground truth"))
    pseudo_label = str(get_in(cfg, "plot.pseudolabel_label", "Pseudolabel"))
    wer_variant = str(get_in(cfg, "plot.wer_variant", "norm"))
    if wer_variant not in {"norm", "raw"}:
        raise ValueError("plot.wer_variant must be one of: norm, raw")

    out_dir = resolve_path(get_in(cfg, "plot.out_dir", "runs/compare_plots"), config_path)
    if out_dir is None:
        raise ValueError("plot.out_dir is required")

    if dry_run:
        summary = {
            "ground_truth_run": str(gt_run),
            "pseudolabel_run": str(pseudo_run),
            "ground_truth_label": gt_label,
            "pseudolabel_label": pseudo_label,
            "wer_variant": wer_variant,
            "out_dir": str(out_dir),
        }
        print("Dry run: resolved plot summary")
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        return summary

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        raise RuntimeError("matplotlib is required. Install it with `pip install matplotlib`.") from exc

    out_dir.mkdir(parents=True, exist_ok=True)

    gt = load_run(gt_run, wer_variant)
    pseudo = load_run(pseudo_run, wer_variant)

    warnings: List[str] = []
    generated_plots: Dict[str, str] = {}

    if gt["dev_wer"] or pseudo["dev_wer"]:
        dev_plot = out_dir / "dev_wer_vs_global_step.png"
        plot_line(
            plt=plt,
            out_path=dev_plot,
            title=f"Dev WER ({wer_variant}) vs Global Step",
            ylabel="WER",
            gt_label=gt_label,
            gt_data=gt["dev_wer"],
            pseudo_label=pseudo_label,
            pseudo_data=pseudo["dev_wer"],
        )
        generated_plots["dev_wer_vs_global_step"] = str(dev_plot)

        if gt["dev_wer"] and pseudo["dev_wer"]:
            try:
                gap_plot = out_dir / "dev_wer_gap_vs_global_step.png"
                plot_dev_gap(
                    plt=plt,
                    out_path=gap_plot,
                    gt_data=gt["dev_wer"],
                    pseudo_data=pseudo["dev_wer"],
                    gt_label=gt_label,
                    pseudo_label=pseudo_label,
                )
                generated_plots["dev_wer_gap_vs_global_step"] = str(gap_plot)
            except ValueError as exc:
                warnings.append(f"Skipped dev WER gap plot: {exc}")
    else:
        warnings.append(
            "Skipped dev WER curve: no eval WER points found in either run "
            "(likely eval did not trigger before training ended)."
        )

    if gt["train_loss"] or pseudo["train_loss"]:
        loss_plot = out_dir / "train_loss_vs_global_step.png"
        plot_line(
            plt=plt,
            out_path=loss_plot,
            title="Train Loss vs Global Step",
            ylabel="CTC Loss",
            gt_label=gt_label,
            gt_data=gt["train_loss"],
            pseudo_label=pseudo_label,
            pseudo_data=pseudo["train_loss"],
        )
        generated_plots["train_loss_vs_global_step"] = str(loss_plot)
    else:
        warnings.append("Skipped train loss curve: no loss points found in either run.")

    test_plot = out_dir / f"final_test_wer_{wer_variant}.png"
    plot_test_bar(
        plt=plt,
        out_path=test_plot,
        gt_label=gt_label,
        gt_wer=gt["test_wer"],
        pseudo_label=pseudo_label,
        pseudo_wer=pseudo["test_wer"],
    )
    generated_plots["final_test_wer"] = str(test_plot)

    summary = {
        "ground_truth_run": str(gt_run),
        "pseudolabel_run": str(pseudo_run),
        "ground_truth": gt,
        "pseudolabel": pseudo,
        "plots": generated_plots,
        "warnings": warnings,
    }
    summary_path = out_dir / "plot_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("Wrote plots:")
    for _, path in generated_plots.items():
        print(f"- {path}")
    if warnings:
        print("Warnings:")
        for warning in warnings:
            print(f"- {warning}")

    print(f"Wrote summary: {summary_path}")
    print(f"Ground-truth test WER ({gt['test_wer_source']}): {gt['test_wer']}")
    print(f"Pseudolabel test WER ({pseudo['test_wer_source']}): {pseudo['test_wer']}")
    return summary
