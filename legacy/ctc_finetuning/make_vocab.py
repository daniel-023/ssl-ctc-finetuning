#!/usr/bin/env python3
import json
import re
import sys
from pathlib import Path

from datasets import load_dataset

# Remove any tags like <v-noise>, <unk>, <laugh>, <music>, etc.
TAG_RE = re.compile(r"<[^>]+>")
# Optional punctuation removal (tune to your needs)
PUNC_RE = re.compile(r"[\,\?\.\!\-\;\:\“\”\"\%\—\–\…\(\)\[\]\{\}]")


def normalize_text_basic(t: str) -> str:
    t = str(t or "").lower()
    t = TAG_RE.sub(" ", t)  # drop tags
    t = PUNC_RE.sub("", t)  # drop punctuation
    t = re.sub(r"\s+", " ", t).strip()
    return t


def build_text_normalizer(yaml_path: str | None):
    file_path = Path(__file__).resolve()
    project_root = None
    for base in [file_path.parent] + list(file_path.parents):
        if (base / "text_normalisation" / "normalizer.py").exists():
            project_root = base
            break
    if project_root is None:
        raise FileNotFoundError(
            "Could not find text_normalisation/normalizer.py from script location. "
            "Ensure text_normalisation is synced next to ctc_finetuning."
        )
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    from text_normalisation.normalizer import NormalizerConfig, Normalizer

    if yaml_path is None:
        yaml_path = str(project_root / "text_normalisation" / "fillers.yaml")
    cfg = NormalizerConfig(yaml_path)
    return Normalizer(cfg), yaml_path


def load_source(
    dataset: str | None,
    config: str | None,
    split: str,
    jsonl: str | None,
):
    if bool(dataset) == bool(jsonl):
        raise ValueError("Provide exactly one of --dataset or --jsonl.")

    if jsonl:
        ds = load_dataset("json", data_files=jsonl, split="train")
        label = f"jsonl={jsonl}"
    else:
        ds = load_dataset(dataset, config, split=split)
        label = f"dataset={dataset} split={split}"

    return ds, label


def apply_score_filter(ds, score_col: str, min_score: float, label: str):
    if min_score <= 0:
        return ds
    if score_col in ds.column_names:
        before = len(ds)
        ds = ds.filter(lambda ex: ex[score_col] is not None and float(ex[score_col]) >= min_score)
        print(f"Score filter [{label}]: kept {len(ds)}/{before} with {score_col}>={min_score}")
    else:
        print(f"Warning: score column '{score_col}' not found in [{label}]; skipping min_score filtering")
    return ds


def collect_vocab(ds, text_col: str, label: str, text_norm_fn):
    if text_col not in ds.column_names:
        raise ValueError(
            f"Text column '{text_col}' not found in [{label}]. "
            f"Available columns: {ds.column_names}"
        )

    vocab = set()
    empty = 0
    for ex in ds:
        txt = text_norm_fn(ex[text_col])
        if not txt:
            empty += 1
            continue
        vocab.update(list(txt))
    return vocab, empty, len(ds)


def main(
    dataset: str | None,
    config: str | None,
    train_split: str,
    text_col: str,
    out_path: str,
    jsonl: str | None,
    score_col: str,
    min_score: float,
    aux_jsonl: str | None,
    aux_text_col: str | None,
    aux_score_col: str,
    aux_min_score: float,
    use_text_normalizer: bool,
    normalizer_yaml: str | None,
):
    text_normalizer = None
    if use_text_normalizer:
        text_normalizer, used_yaml = build_text_normalizer(normalizer_yaml)
        print(f"Using external normalizer in vocab build: {used_yaml}")
    else:
        print("Using basic normalization in vocab build.")

    def text_norm_fn(s: str) -> str:
        if text_normalizer is None:
            return normalize_text_basic(s)
        return text_normalizer.normalize(s)["text_norm"]

    ds, src_label = load_source(dataset, config, train_split, jsonl)
    ds = apply_score_filter(ds, score_col, min_score, src_label)

    vocab, empty, size = collect_vocab(ds, text_col, src_label, text_norm_fn)
    print(f"Loaded [{src_label}] size={size} empty_after_norm={empty}")

    if aux_jsonl:
        aux_ds = load_dataset("json", data_files=aux_jsonl, split="train")
        aux_label = f"aux_jsonl={aux_jsonl}"
        aux_ds = apply_score_filter(aux_ds, aux_score_col, aux_min_score, aux_label)
        aux_col = aux_text_col if aux_text_col else text_col
        aux_vocab, aux_empty, aux_size = collect_vocab(aux_ds, aux_col, aux_label, text_norm_fn)
        print(f"Loaded [{aux_label}] size={aux_size} empty_after_norm={aux_empty}")
        vocab.update(aux_vocab)

    # CTC convention: use '|' to represent space
    vocab.discard(" ")
    vocab_list = sorted(vocab)

    vocab_dict = {c: i for i, c in enumerate(vocab_list)}
    vocab_dict["|"] = len(vocab_dict)
    vocab_dict["[UNK]"] = len(vocab_dict)
    vocab_dict["[PAD]"] = len(vocab_dict)

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(vocab_dict, ensure_ascii=False), encoding="utf-8")

    if aux_jsonl:
        print(f"Merged source: {src_label} + {aux_jsonl}")
    else:
        print(f"Single source: {src_label}")
    print(f"Wrote vocab size={len(vocab_dict)} -> {out}")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=None, help="HF dataset name")
    ap.add_argument("--config", default=None, help="HF dataset config name (or omit)")
    ap.add_argument("--train_split", default="train")
    ap.add_argument("--jsonl", default=None, help="Path to JSONL with transcript column")
    ap.add_argument("--text_col", default="sentence")
    ap.add_argument("--score_col", default="score")
    ap.add_argument("--min_score", type=float, default=0.0)
    ap.add_argument("--aux_jsonl", default=None, help="Optional second JSONL source to merge into vocab")
    ap.add_argument("--aux_text_col", default=None)
    ap.add_argument("--aux_score_col", default="score")
    ap.add_argument("--aux_min_score", type=float, default=0.0)
    ap.add_argument(
        "--use_text_normalizer",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use text_normalisation/normalizer.py for transcript normalization.",
    )
    ap.add_argument("--normalizer_yaml", default=None, help="Path to fillers.yaml for external normalizer.")
    ap.add_argument("--out", default="artifacts/vocab/vocab.json")
    args = ap.parse_args()

    main(
        args.dataset,
        args.config,
        args.train_split,
        args.text_col,
        args.out,
        args.jsonl,
        args.score_col,
        args.min_score,
        args.aux_jsonl,
        args.aux_text_col,
        args.aux_score_col,
        args.aux_min_score,
        args.use_text_normalizer,
        args.normalizer_yaml,
    )
