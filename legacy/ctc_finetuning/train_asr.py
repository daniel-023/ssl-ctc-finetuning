#!/usr/bin/env python3
import json
import re
import sys
import numpy as np
import torch
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Union, Optional

from datasets import load_dataset, Audio, concatenate_datasets
from transformers import (
    Wav2Vec2CTCTokenizer,
    Wav2Vec2FeatureExtractor,
    Wav2Vec2Processor,
    Wav2Vec2ForCTC,
    TrainingArguments,
    Trainer,
)
import evaluate

TAG_RE = re.compile(r"<[^>]+>")
PUNC_RE = re.compile(r"[\,\?\.\!\-\;\:\“\”\"\%\—\–\…\(\)\[\]\{\}]")

def normalize_text_basic(t: str) -> str:
    t = t.lower()
    t = TAG_RE.sub(" ", t)
    t = PUNC_RE.sub("", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t

def build_text_normalizer(yaml_path: Optional[str]):
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

def normalize_batch(batch, text_col: str, text_normalize_fn):
    raw_text = str(batch[text_col] or "")
    batch["raw_text"] = raw_text
    batch["text"] = text_normalize_fn(raw_text)
    return batch

def build_processor(vocab_json: str):
    tokenizer = Wav2Vec2CTCTokenizer(
        vocab_json,
        unk_token="[UNK]",
        pad_token="[PAD]",
        word_delimiter_token="|",
    )
    feature_extractor = Wav2Vec2FeatureExtractor(
        feature_size=1,
        sampling_rate=16000,
        padding_value=0.0,
        do_normalize=True,
        return_attention_mask=True,
    )
    return Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)

TARGET_SR = 16000

def load_audio_16k(audio_obj):
    """
    audio_obj is a dict with decoded audio: {"array": ..., "sampling_rate": ...}
    """
    wav = audio_obj["array"]
    sr = audio_obj["sampling_rate"]

    # convert to mono if needed
    if wav.ndim == 2:
        wav = wav.mean(axis=1)

    if sr != TARGET_SR:
        import librosa
        wav = librosa.resample(wav, orig_sr=sr, target_sr=TARGET_SR)

    return wav

@dataclass
class DataCollatorCTCWithPadding:
    processor: Wav2Vec2Processor
    audio_col: str
    padding: Union[bool, str] = True

    def __call__(self, features: List[Dict]) -> Dict[str, torch.Tensor]:
        # Batch process audio
        wavs = [load_audio_16k(f[self.audio_col]) for f in features]
        batch = self.processor.feature_extractor(
            wavs, sampling_rate=TARGET_SR, return_tensors="pt", padding=self.padding
        )

        # Batch process labels
        texts = [f["text"] for f in features]
        labels_batch = self.processor.tokenizer(
            texts, padding=self.padding, return_tensors="pt"
        )

        batch["labels"] = labels_batch["input_ids"].masked_fill(
            labels_batch["attention_mask"].ne(1), -100
        )
        return batch

def split_exists(dataset: str, config: Optional[str], split: str) -> bool:
    try:
        _ = load_dataset(dataset, config, split=split)
        return True
    except Exception:
        return False

def keep_max_duration(ex, audio_col: str, max_sec: float) -> bool:
    audio = ex[audio_col]
    return (len(audio["array"]) / float(audio["sampling_rate"])) <= max_sec

def load_pseudo_jsonl(args, jsonl_path: str):
    ds = load_dataset("json", data_files=jsonl_path, split="train")

    if args.pseudo_min_score > 0:
        if args.pseudo_score_col in ds.column_names:
            before = len(ds)
            ds = ds.filter(
                lambda ex: ex[args.pseudo_score_col] is not None
                and float(ex[args.pseudo_score_col]) >= args.pseudo_min_score
            )
            print(
                f"Pseudo score filter: kept {len(ds)}/{before} with "
                f"{args.pseudo_score_col}>={args.pseudo_min_score}"
            )
        else:
            print(
                f"Warning: score column '{args.pseudo_score_col}' not found; "
                "skipping pseudo_min_score filtering."
            )

    if args.pseudo_audio_col not in ds.column_names:
        raise ValueError(
            f"Pseudo audio column '{args.pseudo_audio_col}' not found in {jsonl_path}. "
            f"Available columns: {ds.column_names}"
        )
    if args.text_col not in ds.column_names:
        raise ValueError(
            f"Text column '{args.text_col}' not found in {jsonl_path}. "
            f"Available columns: {ds.column_names}"
        )

    if args.pseudo_join_on_hf_id:
        ds = ds.map(
            lambda ex: {"__hf_id": Path(str(ex[args.pseudo_audio_col] or "")).stem},
            num_proc=args.num_proc,
        )

        split_names = [s.strip() for s in args.pseudo_hf_splits.split(",") if s.strip()]
        if (
            args.disallow_test_split_in_pseudo_join
            and args.test_split in set(split_names)
        ):
            raise ValueError(
                "Pseudo/HF id join is configured to include the test split "
                f"('{args.test_split}') in --pseudo_hf_splits={split_names}. "
                "This can cause train/test leakage. Remove the test split from "
                "--pseudo_hf_splits or pass --no-disallow_test_split_in_pseudo_join "
                "to override intentionally."
            )
        hf_parts = []
        for split_name in split_names:
            if split_exists(args.dataset, args.config, split_name):
                hf_parts.append(load_dataset(args.dataset, args.config, split=split_name))
            else:
                print(f"Pseudo join: skip missing HF split '{split_name}'")
        if not hf_parts:
            raise ValueError(
                f"No HF splits available for pseudo join: {split_names}. "
                "Check --pseudo_hf_splits / --dataset / --config."
            )
        hf_ds = hf_parts[0] if len(hf_parts) == 1 else concatenate_datasets(hf_parts)

        if args.pseudo_hf_id_col not in hf_ds.column_names:
            raise ValueError(
                f"HF id column '{args.pseudo_hf_id_col}' not found. "
                f"Available columns: {hf_ds.column_names}"
            )
        if args.audio_col not in hf_ds.column_names:
            raise ValueError(
                f"HF audio column '{args.audio_col}' not found. "
                f"Available columns: {hf_ds.column_names}"
            )

        hf_ids = [str(x) for x in hf_ds[args.pseudo_hf_id_col]]
        id_to_idx = {}
        for i, key in enumerate(hf_ids):
            if key not in id_to_idx:
                id_to_idx[key] = i

        pseudo_ids = [str(x) for x in ds["__hf_id"]]
        pseudo_texts = [str(x) for x in ds[args.text_col]]
        pseudo_scores = [x for x in ds[args.pseudo_score_col]] if args.pseudo_score_col in ds.column_names else None

        keep_hf_indices = []
        keep_texts = []
        keep_scores = [] if pseudo_scores is not None else None
        dropped = 0
        for i, key in enumerate(pseudo_ids):
            idx = id_to_idx.get(key)
            if idx is None:
                dropped += 1
                continue
            keep_hf_indices.append(idx)
            keep_texts.append(pseudo_texts[i])
            if keep_scores is not None:
                keep_scores.append(pseudo_scores[i])

        if not keep_hf_indices:
            raise ValueError(
                "Pseudo/HF id join produced 0 matches. "
                "Check pseudo audio_path format and HF id column."
            )

        out_ds = hf_ds.select(keep_hf_indices)
        if args.text_col in out_ds.column_names:
            out_ds = out_ds.remove_columns(args.text_col)
        out_ds = out_ds.add_column(args.text_col, keep_texts)
        if keep_scores is not None and args.pseudo_score_col not in out_ds.column_names:
            out_ds = out_ds.add_column(args.pseudo_score_col, keep_scores)

        print(
            f"Pseudo join on HF id: matched={len(out_ds)} "
            f"dropped_unmatched={dropped} from pseudo_rows={len(ds)}"
        )
        return out_ds

    raise ValueError(
        "Legacy pseudo local-audio path mode is no longer supported. "
        "Use --pseudo_join_on_hf_id so pseudolabel rows are mapped to HF dataset audio by id."
    )

def main(args):
    processor = build_processor(args.vocab)
    text_normalizer = None
    if args.use_text_normalizer:
        text_normalizer, normalizer_yaml = build_text_normalizer(args.normalizer_yaml)
        print(f"Using external normalizer: {normalizer_yaml}")
    else:
        normalizer_yaml = None
        print("Using basic normalization (legacy regex cleanup).")

    def text_normalize_fn(s: str) -> str:
        if text_normalizer is None:
            return normalize_text_basic(s)
        return text_normalizer.normalize(s)["text_norm"] # type: ignore

    # 1) Load train source
    if args.train_source == "hf":
        train_ds_full = load_dataset(args.dataset, args.config, split=args.train_split)
    elif args.train_source == "pseudo":
        if not args.pseudo_jsonl:
            raise ValueError("--pseudo_jsonl is required when --train_source pseudo")
        train_ds_full = load_pseudo_jsonl(args, args.pseudo_jsonl)
    else:
        raise ValueError(f"Unsupported train_source={args.train_source}")

    # 2) Load test split (final evaluation only; from HF dataset)
    test_ds = load_dataset(args.dataset, args.config, split=args.test_split)

    # 3) Build dev split
    if args.train_source == "hf" and split_exists(args.dataset, args.config, args.val_split):
        train_ds = train_ds_full
        dev_ds = load_dataset(args.dataset, args.config, split=args.val_split)
        print(f"Using existing validation split from HF: {args.val_split}")
    elif args.train_source == "pseudo" and args.pseudo_dev_jsonl:
        train_ds = train_ds_full
        dev_ds = load_pseudo_jsonl(args, args.pseudo_dev_jsonl)
        print(f"Using pseudo dev JSONL: {args.pseudo_dev_jsonl}")
    else:
        split_ds = train_ds_full.train_test_split(test_size=args.val_size, seed=args.seed)
        train_ds = split_ds["train"]
        dev_ds = split_ds["test"]
        print(
            f"No '{args.val_split}' split found; created train/dev split from "
            f"'{args.train_split}' with val_size={args.val_size}"
        )

    # 4) Normalize transcripts -> "text"
    test_text_col = args.test_text_col if args.test_text_col else args.text_col
    map_num_proc = args.num_proc if not args.use_text_normalizer else 1
    train_ds = train_ds.map(
        lambda b: normalize_batch(b, args.text_col, text_normalize_fn),
        num_proc=map_num_proc,
    )
    dev_ds = dev_ds.map(
        lambda b: normalize_batch(b, args.text_col, text_normalize_fn),
        num_proc=map_num_proc,
    )
    test_ds = test_ds.map(
        lambda b: normalize_batch(b, test_text_col, text_normalize_fn),
        num_proc=map_num_proc,
    )

    # 5) Filter out empty transcripts after tag/punc stripping
    train_ds = train_ds.filter(lambda x: len(x["text"]) > 0)
    dev_ds = dev_ds.filter(lambda x: len(x["text"]) > 0)
    test_ds = test_ds.filter(lambda x: len(x["text"]) > 0)

    # 6) Decode audio at target sample rate
    if args.audio_col not in test_ds.column_names and args.test_audio_col in test_ds.column_names:
        test_ds = test_ds.rename_column(args.test_audio_col, args.audio_col)

    if args.audio_col not in train_ds.column_names or args.audio_col not in dev_ds.column_names:
        raise ValueError(
            f"audio_col='{args.audio_col}' not found in training/dev datasets. "
            "For pseudo mode, set --pseudo_audio_col and --audio_col correctly."
        )
    if args.audio_col not in test_ds.column_names:
        raise ValueError(
            f"audio_col='{args.audio_col}' not found in test dataset. "
            f"Consider setting --test_audio_col (current: {args.test_audio_col})."
        )

    train_ds = train_ds.cast_column(args.audio_col, Audio(sampling_rate=TARGET_SR))
    dev_ds = dev_ds.cast_column(args.audio_col, Audio(sampling_rate=TARGET_SR))
    test_ds = test_ds.cast_column(args.audio_col, Audio(sampling_rate=TARGET_SR))

    # 7) Enforce max audio duration
    if args.max_sec > 0:
        train_ds = train_ds.filter(
            lambda ex: keep_max_duration(ex, args.audio_col, args.max_sec),
            num_proc=args.num_proc,
        )
        dev_ds = dev_ds.filter(
            lambda ex: keep_max_duration(ex, args.audio_col, args.max_sec),
            num_proc=args.num_proc,
        )
        test_ds = test_ds.filter(
            lambda ex: keep_max_duration(ex, args.audio_col, args.max_sec),
            num_proc=args.num_proc,
        )

    # 8) Keep only required columns
    train_ds = train_ds.select_columns([args.audio_col, "text"])
    dev_ds = dev_ds.select_columns([args.audio_col, "text"])
    test_ds = test_ds.select_columns([args.audio_col, "text", "raw_text"])

    data_collator = DataCollatorCTCWithPadding(processor=processor, audio_col=args.audio_col)
    wer = evaluate.load("wer")

    def compute_metrics(pred):
        pred_ids = np.argmax(pred.predictions, axis=-1)

        label_ids = pred.label_ids.copy()
        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id

        pred_str = processor.batch_decode(pred_ids)
        label_str = processor.batch_decode(label_ids, group_tokens=False)

        pred_norm = [text_normalize_fn(s) for s in pred_str]
        label_norm = [text_normalize_fn(s) for s in label_str]

        label_ref_wer = wer.compute(predictions=pred_str, references=label_str)
        return {
            "wer_decoded_label_ref": label_ref_wer,
            # Backward-compatible alias for older tooling.
            "wer_decoded_raw": label_ref_wer,
            "wer_decoded_norm": wer.compute(predictions=pred_norm, references=label_norm),
        }

    # 9) Model
    model = Wav2Vec2ForCTC.from_pretrained(
        "facebook/wav2vec2-xls-r-300m",
        pad_token_id=processor.tokenizer.pad_token_id,
        vocab_size=len(processor.tokenizer),
        ctc_loss_reduction="mean",
    )

    # Full model fine-tuning: train encoder and lm_head
    for p in model.wav2vec2.parameters():
        p.requires_grad = True
    for p in model.lm_head.parameters():
        p.requires_grad = True

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(
        f"Source={args.train_source}; Splits: train={len(train_ds)} "
        f"eval(dev)={len(dev_ds)} test={len(test_ds)}; trainable_params={trainable}"
    )

    # 10) Training args
    targs = TrainingArguments(
        output_dir=args.out,
        fp16=True,
        per_device_train_batch_size=args.bs,
        per_device_eval_batch_size=args.bs,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type=args.lr_scheduler,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        num_train_epochs=args.epochs,
        group_by_length=False,
        eval_strategy="steps",
        save_strategy="steps",
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        load_best_model_at_end=True,
        metric_for_best_model=args.best_metric,
        greater_is_better=False,
        logging_steps=args.logging_steps,
        save_total_limit=args.save_total_limit,
        max_grad_norm=args.max_grad_norm,
        report_to="none",
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=targs,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        train_dataset=train_ds,
        eval_dataset=dev_ds,
    )

    trainer.train()
    trainer.save_model(args.out)
    processor.save_pretrained(args.out)

    # 11) Final: evaluate ONCE on TEST
    print("\nEvaluating on TEST split...")
    pred_out = trainer.predict(test_ds, metric_key_prefix="test")
    metrics = dict(pred_out.metrics)

    pred_ids = np.argmax(pred_out.predictions, axis=-1)
    pred_text = processor.batch_decode(pred_ids)
    test_refs_raw = [str(x) for x in test_ds["raw_text"]]
    test_refs_norm = [str(x) for x in test_ds["text"]]
    pred_text_norm = [text_normalize_fn(s) for s in pred_text]

    metrics["test_wer_raw_ref"] = wer.compute(predictions=pred_text, references=test_refs_raw)
    metrics["test_wer_norm_ref"] = wer.compute(predictions=pred_text_norm, references=test_refs_norm)

    print("TEST metrics:", metrics)
    metrics_path = Path(args.out) / "test_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Wrote TEST metrics -> {metrics_path}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()

    ap.add_argument("--dataset", default="pengyizhou/nsc-imda-part6")
    ap.add_argument("--config", default=None)
    ap.add_argument("--train_source", choices=["hf", "pseudo"], default="hf")
    ap.add_argument(
        "--use_text_normalizer",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use text_normalisation/normalizer.py for transcript normalization.",
    )
    ap.add_argument("--normalizer_yaml", default=None, help="Path to fillers.yaml for external normalizer.")

    ap.add_argument("--audio_col", default="audio")
    ap.add_argument("--text_col", default="sentence")
    ap.add_argument("--test_text_col", default=None)
    ap.add_argument("--test_audio_col", default="audio")

    ap.add_argument("--train_split", default="train")
    ap.add_argument("--val_split", default="validation")
    ap.add_argument("--test_split", default="test")
    ap.add_argument("--val_size", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--pseudo_jsonl", default=None)
    ap.add_argument("--pseudo_dev_jsonl", default=None)
    ap.add_argument("--pseudo_audio_col", default="audio_path")
    ap.add_argument(
        "--pseudo_join_on_hf_id",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Map pseudo rows to HF dataset audio by id derived from pseudo_audio_col stem.",
    )
    ap.add_argument("--pseudo_hf_id_col", default="id")
    ap.add_argument(
        "--pseudo_hf_splits",
        default="train,validation",
        help="Comma-separated HF splits used as audio source when pseudo_join_on_hf_id is enabled.",
    )
    ap.add_argument(
        "--disallow_test_split_in_pseudo_join",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Safety guard: error if --test_split appears in --pseudo_hf_splits to avoid "
            "train/test leakage in pseudolabel join."
        ),
    )
    ap.add_argument("--pseudo_score_col", default="score")
    ap.add_argument("--pseudo_min_score", type=float, default=0.0)

    ap.add_argument("--vocab", required=True)
    ap.add_argument("--out", required=True)

    ap.add_argument("--max_sec", type=float, default=30.0)
    ap.add_argument("--num_proc", type=int, default=4)

    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--grad_accum", type=int, default=1)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--epochs", type=float, default=20.0)
    ap.add_argument("--lr_scheduler", default="cosine")
    ap.add_argument("--warmup_ratio", type=float, default=0.1)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--max_grad_norm", type=float, default=1.0)

    ap.add_argument("--eval_steps", type=int, default=1200)
    ap.add_argument("--save_steps", type=int, default=1200)
    ap.add_argument("--logging_steps", type=int, default=50)
    ap.add_argument("--save_total_limit", type=int, default=4)
    ap.add_argument(
        "--best_metric",
        choices=["wer_decoded_label_ref", "wer_decoded_raw", "wer_decoded_norm"],
        default="wer_decoded_norm",
        help="Metric used by load_best_model_at_end.",
    )

    args = ap.parse_args()
    main(args)
