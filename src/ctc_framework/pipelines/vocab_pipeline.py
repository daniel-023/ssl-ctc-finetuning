import json
from pathlib import Path
from typing import Optional

from datasets import load_dataset

from ctc_framework.config.loader import get_in
from ctc_framework.pipelines.common import build_text_normalizer, normalize_text_basic, resolve_path


def _load_source(cfg: dict, config_path: Path):
    backend = str(get_in(cfg, "dataset.backend", "hf"))
    transcript_source = str(get_in(cfg, "transcript.source", "dataset_field"))

    if transcript_source == "external_gt_json":
        json_path = resolve_path(get_in(cfg, "transcript.external_gt.json_path"), config_path)
        if json_path is None or not json_path.exists():
            raise FileNotFoundError(f"External transcript JSON not found: {json_path}")
        text_col = str(get_in(cfg, "transcript.external_gt.text_col", "text"))
        ds = load_dataset("json", data_files=str(json_path), split="train")
        label = f"external_gt_json={json_path}"
        return ds, text_col, label

    if backend == "local":
        manifest = resolve_path(get_in(cfg, "dataset.local.manifests.train"), config_path)
        if manifest is None or not manifest.exists():
            raise FileNotFoundError(f"Local train manifest not found: {manifest}")
        text_col = str(get_in(cfg, "dataset.columns.transcript"))
        ds = load_dataset("json", data_files=str(manifest), split="train")
        label = f"local_manifest={manifest}"
        return ds, text_col, label

    hf_name = str(get_in(cfg, "dataset.hf_name"))
    hf_config = get_in(cfg, "dataset.hf_config")
    train_split = str(get_in(cfg, "dataset.splits.train", "train"))
    text_col = str(get_in(cfg, "dataset.columns.transcript"))
    ds = load_dataset(hf_name, hf_config, split=train_split)
    label = f"dataset={hf_name} split={train_split}"
    return ds, text_col, label


def _apply_score_filter(ds, score_col: str, min_score: float, label: str):
    if min_score <= 0:
        return ds
    if score_col in ds.column_names:
        before = len(ds)
        ds = ds.filter(lambda ex: ex[score_col] is not None and float(ex[score_col]) >= min_score)
        print(f"Score filter [{label}]: kept {len(ds)}/{before} with {score_col}>={min_score}")
    else:
        print(f"Warning: score column '{score_col}' not found in [{label}]; skipping min_score filtering")
    return ds


def _collect_vocab(ds, text_col: str, label: str, text_norm_fn):
    if text_col not in ds.column_names:
        raise ValueError(
            f"Text column '{text_col}' not found in [{label}]. Available columns: {ds.column_names}"
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


def run_vocab_build(cfg: dict, config_path: Path, dry_run: bool = False):
    out_path = resolve_path(get_in(cfg, "vocab.out_path"), config_path)
    if out_path is None:
        raise ValueError("vocab.out_path is required")

    use_text_normalizer = bool(get_in(cfg, "normalization.use_text_normalizer", True))
    normalizer_yaml = get_in(cfg, "normalization.normalizer_yaml")

    if use_text_normalizer:
        text_normalizer, used_yaml = build_text_normalizer(normalizer_yaml, config_path)
        print(f"Using external normalizer in vocab build: {used_yaml}")
    else:
        text_normalizer = None
        used_yaml = None
        print("Using basic normalization in vocab build.")

    def text_norm_fn(s: str) -> str:
        if text_normalizer is None:
            return normalize_text_basic(s)
        return text_normalizer.normalize(s)["text_norm"]

    ds, text_col, src_label = _load_source(cfg, config_path)
    ds = _apply_score_filter(
        ds,
        str(get_in(cfg, "vocab.score_col", "score")),
        float(get_in(cfg, "vocab.min_score", 0.0)),
        src_label,
    )

    vocab, empty, size = _collect_vocab(ds, text_col, src_label, text_norm_fn)
    print(f"Loaded [{src_label}] size={size} empty_after_norm={empty}")

    mode = str(get_in(cfg, "vocab.mode", "dataset_only"))
    if mode == "shared_hf_plus_pseudolabel":
        pseudo_json = resolve_path(get_in(cfg, "transcript.pseudolabel.json_path"), config_path)
        if pseudo_json is None or not pseudo_json.exists():
            raise FileNotFoundError(f"Pseudolabel JSON not found for aux vocab merge: {pseudo_json}")

        aux_ds = load_dataset("json", data_files=str(pseudo_json), split="train")
        aux_label = f"pseudolabel_json={pseudo_json}"
        aux_ds = _apply_score_filter(
            aux_ds,
            str(get_in(cfg, "vocab.aux_score_col", "score")),
            float(get_in(cfg, "vocab.aux_min_score", get_in(cfg, "transcript.pseudolabel.min_score", 0.0))),
            aux_label,
        )
        aux_text_col = text_col
        aux_vocab, aux_empty, aux_size = _collect_vocab(aux_ds, aux_text_col, aux_label, text_norm_fn)
        print(f"Loaded [{aux_label}] size={aux_size} empty_after_norm={aux_empty}")
        vocab.update(aux_vocab)

    vocab.discard(" ")
    vocab_list = sorted(vocab)

    vocab_dict = {c: i for i, c in enumerate(vocab_list)}
    vocab_dict["|"] = len(vocab_dict)
    vocab_dict["[UNK]"] = len(vocab_dict)
    vocab_dict["[PAD]"] = len(vocab_dict)

    summary = {
        "source": src_label,
        "normalizer_enabled": use_text_normalizer,
        "normalizer_yaml": used_yaml,
        "vocab_size": len(vocab_dict),
        "output_path": str(out_path),
    }

    if dry_run:
        print("Dry run: resolved vocab build summary")
        print(json.dumps(summary, indent=2))
        return summary

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(vocab_dict, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote vocab size={len(vocab_dict)} -> {out_path}")

    (out_path.parent / "vocab_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return summary
