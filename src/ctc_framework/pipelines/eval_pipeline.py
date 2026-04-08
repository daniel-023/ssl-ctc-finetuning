import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import evaluate
import torch
from datasets import Audio, load_dataset

from ctc_framework.config.loader import get_in
from ctc_framework.pipelines.utils import (
    TARGET_SR,
    apply_jsonl_transcripts,
    build_normalizer_fn,
    keep_max_duration,
    load_audio_16k,
    load_jsonl_transcript_map,
    maybe_prefix_local_audio_paths,
    resolve_path,
)

DIGIT_RE = re.compile(r"\d")


@dataclass
class InferenceCollator:
    processor: Any
    audio_col: str

    def __call__(self, features: List[Dict]) -> Dict[str, torch.Tensor]:
        wavs = [load_audio_16k(f[self.audio_col]) for f in features]
        return self.processor.feature_extractor(
            wavs,
            sampling_rate=TARGET_SR,
            return_tensors="pt",
            padding=True,
        )


def _resolve_eval_manifest(cfg: dict, eval_split: str, config_path: Path) -> Path:
    explicit = get_in(cfg, "eval.local_manifest")
    if explicit:
        p = resolve_path(explicit, config_path)
        if p is None or not p.exists():
            raise FileNotFoundError(f"eval.local_manifest not found: {p}")
        return p

    split_train = str(get_in(cfg, "dataset.splits.train", "train"))
    split_val = str(get_in(cfg, "dataset.splits.val", "validation"))
    split_test = str(get_in(cfg, "dataset.splits.test", "test"))

    if eval_split == split_train:
        selected = get_in(cfg, "dataset.local.manifests.train")
    elif eval_split == split_val:
        selected = get_in(cfg, "dataset.local.manifests.val")
    elif eval_split == split_test:
        selected = get_in(cfg, "dataset.local.manifests.test")
    else:
        selected = (
            get_in(cfg, "dataset.local.manifests.test")
            or get_in(cfg, "dataset.local.manifests.val")
            or get_in(cfg, "dataset.local.manifests.train")
        )

    p = resolve_path(selected, config_path)
    if p is None or not p.exists():
        raise FileNotFoundError(
            "Could not resolve a local eval manifest. "
            "Set eval.local_manifest or dataset.local.manifests.*."
        )
    return p


def _resolve_device(device_cfg: str) -> str:
    if device_cfg == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device_cfg


# ---------------------------------------------------------------------------
# Eval sub-functions
# ---------------------------------------------------------------------------

def _load_eval_dataset(
    cfg: dict,
    config_path: Path,
    backend: str,
    eval_split: str,
    audio_col: str,
    text_col: str,
) -> Tuple[any, str]:
    """Load the evaluation dataset and apply transcript join if configured.

    Returns (ds, dataset_label).
    """
    if backend == "hf":
        hf_name = str(get_in(cfg, "dataset.hf_name"))
        hf_config = get_in(cfg, "dataset.hf_config")
        ds = load_dataset(hf_name, hf_config, split=eval_split)
        dataset_label = f"hf:{hf_name}/{hf_config}:{eval_split}"
    elif backend == "local":
        manifest = _resolve_eval_manifest(cfg, eval_split, config_path)
        ds = load_dataset("json", data_files=str(manifest), split="train")
        local_audio_root = resolve_path(get_in(cfg, "dataset.local.audio_root"), config_path)
        ds = maybe_prefix_local_audio_paths(
            ds,
            audio_col=audio_col,
            audio_root=local_audio_root,
            num_proc=max(1, int(get_in(cfg, "eval.num_workers"))),
        )
        dataset_label = f"local:{manifest}"
    else:
        raise ValueError("dataset.backend must be 'hf' or 'local'")

    transcript_source = str(get_in(cfg, "transcript.source", "inline"))
    transcript_json_type = str(get_in(cfg, "transcript.jsonl.type", "ground_truth"))
    if transcript_source == "jsonl" and transcript_json_type == "ground_truth":
        transcript_map = load_jsonl_transcript_map(cfg, config_path)
        ds = apply_jsonl_transcripts(ds, eval_split, cfg, transcript_map, text_col=text_col)

    if audio_col not in ds.column_names:
        raise ValueError(f"audio column '{audio_col}' not in dataset columns: {ds.column_names}")
    if text_col not in ds.column_names:
        raise ValueError(f"text column '{text_col}' not in dataset columns: {ds.column_names}")

    return ds, dataset_label


def _filter_eval_dataset(
    ds,
    cfg: dict,
    audio_col: str,
    text_col: str,
    dry_run: bool,
) -> any:
    """Cast audio, apply duration/digit/sample-count filters.

    Duration filter is skipped during dry_run (consistent with training behaviour).
    """
    ds = ds.cast_column(audio_col, Audio(sampling_rate=TARGET_SR))

    max_sec = float(get_in(cfg, "eval.max_sec"))
    if max_sec > 0 and not dry_run:
        before = len(ds)
        ds = ds.filter(lambda ex: keep_max_duration(ex, audio_col, max_sec))
        print(f"Duration filter: kept {len(ds)}/{before} with max_sec={max_sec}")

    if bool(get_in(cfg, "eval.discard_number_samples")):
        before = len(ds)
        ds = ds.filter(lambda ex: not DIGIT_RE.search(str(ex[text_col] or "")))
        print(f"Discard-number filter: kept {len(ds)}/{before}")

    num_samples = int(get_in(cfg, "eval.num_samples"))
    if num_samples > 0:
        ds = ds.select(range(min(num_samples, len(ds))))

    return ds


def _run_inference(model, loader, processor, device: str) -> List[str]:
    """Run greedy CTC decoding over the dataloader and return raw predicted strings."""
    pred_raw: List[str] = []
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            logits = model(**batch).logits
            pred_ids = torch.argmax(logits, dim=-1).cpu().numpy()
            pred_raw.extend(processor.batch_decode(pred_ids))
    return pred_raw


def _write_eval_outputs(
    metrics: dict,
    ds,
    id_col: str,
    pred_raw: List[str],
    raw_refs: List[str],
    norm_refs: List[str],
    nofill_refs: List[str],
    pred_norm: List[str],
    pred_nofill: List[str],
    out_json: Path,
    out_jsonl: Optional[Path],
) -> None:
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    print(f"Wrote metrics -> {out_json}")

    if out_jsonl is not None:
        id_values = ds[id_col] if id_col in ds.column_names else [None] * len(ds)
        out_jsonl.parent.mkdir(parents=True, exist_ok=True)
        with out_jsonl.open("w", encoding="utf-8") as fout:
            for i in range(len(ds)):
                row = {
                    "id": id_values[i],
                    "ref_raw": raw_refs[i],
                    "ref_norm": norm_refs[i],
                    "ref_no_fill": nofill_refs[i],
                    "hyp_raw": pred_raw[i],
                    "hyp_norm": pred_norm[i],
                    "hyp_no_fill": pred_nofill[i],
                }
                fout.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"Wrote per-utterance output -> {out_jsonl}")


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run_eval(cfg: dict, config_path: Path, dry_run: bool = False):
    model_dir = resolve_path(get_in(cfg, "eval.model_dir"), config_path)
    if model_dir is None and not dry_run:
        raise ValueError("eval.model_dir is required")
    if model_dir is not None and not model_dir.exists() and not dry_run:
        raise FileNotFoundError(f"Model directory not found: {model_dir}")

    backend = str(get_in(cfg, "dataset.backend", "hf"))
    eval_split = str(get_in(cfg, "eval.split"))

    audio_col = get_in(cfg, "eval.columns.audio") or get_in(cfg, "dataset.columns.audio")
    text_col = get_in(cfg, "eval.columns.transcript") or get_in(cfg, "dataset.columns.transcript")
    id_col = get_in(cfg, "eval.columns.id") or get_in(cfg, "dataset.columns.id", "id")
    if not audio_col or not text_col:
        raise ValueError("Unable to resolve eval audio/transcript columns from config.")
    audio_col = str(audio_col)
    text_col = str(text_col)
    id_col = str(id_col)

    transcript_source = str(get_in(cfg, "transcript.source", "inline"))
    transcript_json_type = str(get_in(cfg, "transcript.jsonl.type", "ground_truth"))

    normalizer_yaml = get_in(cfg, "eval.normalizer_yaml") or get_in(cfg, "normalization.normalizer_yaml")
    space_chinese_chars = bool(get_in(cfg, "eval.space_chinese_chars"))
    verbalize_numbers = bool(get_in(cfg, "eval.verbalize_numbers"))
    device = _resolve_device(str(get_in(cfg, "eval.device")))

    _, normalize_both, used_yaml = build_normalizer_fn(
        normalizer_yaml,
        config_path,
        enable_chinese_spacing=space_chinese_chars,
        verbalize_numbers=verbalize_numbers,
    )

    ds, dataset_label = _load_eval_dataset(cfg, config_path, backend, eval_split, audio_col, text_col)
    ds = _filter_eval_dataset(ds, cfg, audio_col, text_col, dry_run)

    out_json = resolve_path(get_in(cfg, "eval.out_json"), config_path)
    out_jsonl_raw = get_in(cfg, "eval.out_jsonl")
    out_jsonl = resolve_path(out_jsonl_raw, config_path) if out_jsonl_raw else None

    summary = {
        "dataset": dataset_label,
        "backend": backend,
        "split": eval_split,
        "model_dir": str(model_dir) if model_dir is not None else None,
        "num_samples": len(ds),
        "audio_col": audio_col,
        "text_col": text_col,
        "id_col": id_col,
        "transcript_source": transcript_source,
        "transcript_json_type": transcript_json_type,
        "normalizer_yaml": used_yaml,
        "space_chinese_chars": space_chinese_chars,
        "verbalize_numbers": verbalize_numbers,
        "device": device,
        "batch_size": int(get_in(cfg, "eval.batch_size")),
    }

    if dry_run:
        print("Dry run: resolved eval summary")
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        return summary

    from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

    if model_dir is None:
        raise ValueError("eval.model_dir is required")

    processor = Wav2Vec2Processor.from_pretrained(str(model_dir))
    model = Wav2Vec2ForCTC.from_pretrained(str(model_dir)).to(device)
    model.eval()

    collator = InferenceCollator(processor=processor, audio_col=audio_col)
    loader = torch.utils.data.DataLoader(
        ds,
        batch_size=int(get_in(cfg, "eval.batch_size")),
        shuffle=False,
        collate_fn=collator,
        num_workers=int(get_in(cfg, "eval.num_workers")),
        pin_memory=(device == "cuda"),
    )

    pred_raw = _run_inference(model, loader, processor, device)

    raw_refs = [str(x or "").lower() for x in ds[text_col]]
    norm_refs: List[str] = []
    nofill_refs: List[str] = []
    for text in raw_refs:
        norm, no_fill = normalize_both(text)
        norm_refs.append(norm)
        nofill_refs.append(no_fill)

    pred_norm: List[str] = []
    pred_nofill: List[str] = []
    for text in pred_raw:
        norm, no_fill = normalize_both(text)
        pred_norm.append(norm)
        pred_nofill.append(no_fill)

    if len(pred_raw) != len(raw_refs):
        raise RuntimeError(f"Prediction count mismatch: preds={len(pred_raw)} refs={len(raw_refs)}")

    wer = evaluate.load("wer")
    metrics = {
        "dataset": dataset_label,
        "split": eval_split,
        "num_samples": len(raw_refs),
        "model_dir": str(model_dir),
        "normalizer_yaml": used_yaml,
        "wer_raw_ref": wer.compute(predictions=pred_raw, references=raw_refs),
        "wer_norm_ref": wer.compute(predictions=pred_norm, references=norm_refs),
        "wer_norm_no_fill_ref": wer.compute(predictions=pred_nofill, references=nofill_refs),
    }

    if out_json is None:
        raise ValueError("eval.out_json is required")

    _write_eval_outputs(
        metrics, ds, id_col,
        pred_raw, raw_refs, norm_refs, nofill_refs,
        pred_norm, pred_nofill,
        out_json, out_jsonl,
    )

    return metrics
