#!/usr/bin/env python3
import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import evaluate
import torch
from datasets import Audio, load_dataset
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

TAG_RE = re.compile(r"<[^>]+>")
PUNC_RE = re.compile(r"[\,\?\.\!\-\;\:\“\”\"\%\—\–\…\(\)\[\]\{\}]")
DIGIT_RE = re.compile(r"\d")
TARGET_SR = 16000


def normalize_text_basic(t: str) -> str:
    t = str(t or "").lower()
    t = TAG_RE.sub(" ", t)
    t = PUNC_RE.sub("", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def build_text_normalizer(
    yaml_path: Optional[str],
    enable_chinese_spacing: bool = True,
    verbalize_numbers: bool = False,
):
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

    from text_normalisation.normalizer import Normalizer, NormalizerConfig

    if yaml_path is None:
        yaml_path = str(project_root / "text_normalisation" / "fillers.yaml")
    cfg = NormalizerConfig(yaml_path)
    return (
        Normalizer(
            cfg,
            enable_chinese_spacing=enable_chinese_spacing,
            verbalize_numbers=verbalize_numbers,
        ),
        yaml_path,
    )


def load_audio_16k(audio_obj):
    wav = audio_obj["array"]
    sr = audio_obj["sampling_rate"]

    if wav.ndim == 2:
        wav = wav.mean(axis=1)
    if sr != TARGET_SR:
        import librosa

        wav = librosa.resample(wav, orig_sr=sr, target_sr=TARGET_SR)
    return wav


def keep_max_duration(ex, audio_col: str, max_sec: float) -> bool:
    audio = ex[audio_col]
    return (len(audio["array"]) / float(audio["sampling_rate"])) <= max_sec


@dataclass
class InferenceCollator:
    processor: Wav2Vec2Processor
    audio_col: str

    def __call__(self, features: List[Dict]) -> Dict[str, torch.Tensor]:
        wavs = [load_audio_16k(f[self.audio_col]) for f in features]
        return self.processor.feature_extractor(
            wavs,
            sampling_rate=TARGET_SR,
            return_tensors="pt",
            padding=True,
        )


def parse_args():
    ap = argparse.ArgumentParser(
        description=(
            "Run inference with a fine-tuned CTC model on an HF dataset split and "
            "compute WER against raw/original, normalized, and normalized-no-filler refs."
        )
    )
    ap.add_argument("--model_dir", required=True, help="Fine-tuned checkpoint/run directory.")
    ap.add_argument("--dataset", required=True, help="HF dataset name.")
    ap.add_argument("--config", default=None, help="HF dataset config.")
    ap.add_argument("--split", default="test", help="HF split to evaluate, e.g. test/validation.")
    ap.add_argument("--audio_col", default="audio")
    ap.add_argument("--text_col", default="text")
    ap.add_argument("--id_col", default="id", help="Optional id column for per-utterance outputs.")

    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--num_samples", type=int, default=0, help="If >0, evaluate first N samples only.")
    ap.add_argument("--max_sec", type=float, default=0.0, help="If >0, drop utterances longer than max_sec.")
    ap.add_argument("--discard_number_samples", action="store_true")

    ap.add_argument(
        "--use_text_normalizer",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use text_normalisation/normalizer.py for normalization.",
    )
    ap.add_argument("--normalizer_yaml", default=None)
    ap.add_argument(
        "--space_chinese_chars",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Pass through to Normalizer(enable_chinese_spacing=...).",
    )
    ap.add_argument(
        "--verbalize_numbers",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Pass through to Normalizer(verbalize_numbers=...).",
    )

    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--out_json", required=True, help="Path to write aggregate WER metrics JSON.")
    ap.add_argument(
        "--out_jsonl",
        default=None,
        help="Optional per-utterance output JSONL with predictions and normalized variants.",
    )
    return ap.parse_args()


def main():
    args = parse_args()

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    processor = Wav2Vec2Processor.from_pretrained(args.model_dir)
    model = Wav2Vec2ForCTC.from_pretrained(args.model_dir).to(device)
    model.eval()

    ds = load_dataset(args.dataset, args.config, split=args.split)
    if args.audio_col not in ds.column_names:
        raise ValueError(f"audio_col '{args.audio_col}' not in dataset columns: {ds.column_names}")
    if args.text_col not in ds.column_names:
        raise ValueError(f"text_col '{args.text_col}' not in dataset columns: {ds.column_names}")

    ds = ds.cast_column(args.audio_col, Audio(sampling_rate=TARGET_SR))
    if args.max_sec > 0:
        before = len(ds)
        ds = ds.filter(lambda ex: keep_max_duration(ex, args.audio_col, args.max_sec))
        print(f"Duration filter: kept {len(ds)}/{before} with max_sec={args.max_sec}")

    if args.discard_number_samples:
        before = len(ds)
        ds = ds.filter(lambda ex: not DIGIT_RE.search(str(ex[args.text_col] or "")))
        print(f"Discard-number filter: kept {len(ds)}/{before}")

    if args.num_samples > 0:
        ds = ds.select(range(min(args.num_samples, len(ds))))

    text_normalizer = None
    used_yaml = None
    if args.use_text_normalizer:
        text_normalizer, used_yaml = build_text_normalizer(
            args.normalizer_yaml,
            enable_chinese_spacing=args.space_chinese_chars,
            verbalize_numbers=args.verbalize_numbers,
        )
        print(f"Using external normalizer: {used_yaml}")
    else:
        print("Using basic normalization fallback; no dedicated filler removal.")

    def normalize_both(s: str) -> Tuple[str, str]:
        if text_normalizer is None:
            n = normalize_text_basic(s)
            return n, n
        out = text_normalizer.normalize(s)
        return str(out["text_norm"]), str(out["text_no_fill"])

    raw_refs = [str(x or "").lower() for x in ds[args.text_col]]
    norm_refs = []
    nofill_refs = []
    for text in raw_refs:
        n, nf = normalize_both(text)
        norm_refs.append(n)
        nofill_refs.append(nf)

    collator = InferenceCollator(processor=processor, audio_col=args.audio_col)
    loader = torch.utils.data.DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collator,
        num_workers=args.num_workers,
    )

    pred_raw: List[str] = []
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            logits = model(**batch).logits
            pred_ids = torch.argmax(logits, dim=-1).cpu().numpy()
            pred_raw.extend(processor.batch_decode(pred_ids))

    pred_norm = []
    pred_nofill = []
    for text in pred_raw:
        n, nf = normalize_both(text)
        pred_norm.append(n)
        pred_nofill.append(nf)

    if len(pred_raw) != len(raw_refs):
        raise RuntimeError(f"Prediction count mismatch: preds={len(pred_raw)} refs={len(raw_refs)}")

    wer = evaluate.load("wer")
    metrics = {
        "dataset": args.dataset,
        "config": args.config,
        "split": args.split,
        "num_samples": len(raw_refs),
        "model_dir": str(Path(args.model_dir).resolve()),
        "normalizer_enabled": bool(args.use_text_normalizer),
        "normalizer_yaml": used_yaml,
        "wer_raw_ref": wer.compute(predictions=pred_raw, references=raw_refs),
        "wer_norm_ref": wer.compute(predictions=pred_norm, references=norm_refs),
        "wer_norm_no_fill_ref": wer.compute(predictions=pred_nofill, references=nofill_refs),
    }

    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    print(f"Wrote metrics -> {out_json}")

    if args.out_jsonl:
        id_values = ds[args.id_col] if args.id_col in ds.column_names else [None] * len(ds)
        out_jsonl = Path(args.out_jsonl)
        out_jsonl.parent.mkdir(parents=True, exist_ok=True)
        with out_jsonl.open("w", encoding="utf-8") as f:
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
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"Wrote per-utterance output -> {out_jsonl}")


if __name__ == "__main__":
    main()
