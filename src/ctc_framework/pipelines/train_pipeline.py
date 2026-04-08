import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import evaluate
import numpy as np
import torch
from datasets import Audio, concatenate_datasets, load_dataset

from ctc_framework.config.loader import get_in, save_yaml
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


@dataclass
class DataCollatorCTCWithPadding:
    processor: Any
    audio_col: str
    padding: Union[bool, str] = True

    def __call__(self, features: List[Dict]) -> Dict[str, torch.Tensor]:
        wavs = [load_audio_16k(f[self.audio_col]) for f in features]
        batch = self.processor.feature_extractor(
            wavs,
            sampling_rate=TARGET_SR,
            return_tensors="pt",
            padding=self.padding,
        )
        texts = [f["text"] for f in features]
        labels_batch = self.processor.tokenizer(texts, padding=self.padding, return_tensors="pt")
        batch["labels"] = labels_batch["input_ids"].masked_fill(labels_batch["attention_mask"].ne(1), -100)
        return batch


def build_processor(vocab_json: Path):
    from transformers import Wav2Vec2CTCTokenizer, Wav2Vec2FeatureExtractor, Wav2Vec2Processor

    tokenizer = Wav2Vec2CTCTokenizer(
        str(vocab_json),
        unk_token="[UNK]",
        pad_token="[PAD]",
        word_delimiter_token="|",
    )
    feature_extractor = Wav2Vec2FeatureExtractor(
        feature_size=1,
        sampling_rate=TARGET_SR,
        padding_value=0.0,
        do_normalize=True,
        return_attention_mask=True,
    )
    return Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)


def _manifest_for_split(cfg: dict, split: str) -> Optional[Path]:
    split_train = str(get_in(cfg, "dataset.splits.train", "train"))
    split_val = str(get_in(cfg, "dataset.splits.val", "validation"))
    split_test = str(get_in(cfg, "dataset.splits.test", "test"))

    if split == split_train:
        value = get_in(cfg, "dataset.local.manifests.train")
    elif split == split_val:
        value = get_in(cfg, "dataset.local.manifests.val")
    elif split == split_test:
        value = get_in(cfg, "dataset.local.manifests.test")
    else:
        value = None

    if value is None:
        return None
    return Path(str(value))


def _resolve_local_manifest(manifest: Optional[Path], config_path: Path, split: str) -> Path:
    if manifest is None:
        raise ValueError(
            f"No local manifest configured for split '{split}'. "
            "Set dataset.local.manifests.train/val/test."
        )
    p = resolve_path(str(manifest), config_path)
    if p is None or not p.exists():
        raise FileNotFoundError(f"Local manifest for split '{split}' not found: {p}")
    return p


def load_split(cfg: dict, config_path: Path, split: str):
    backend = str(get_in(cfg, "dataset.backend", "hf"))
    if backend == "hf":
        hf_name = str(get_in(cfg, "dataset.hf_name"))
        hf_config = get_in(cfg, "dataset.hf_config")
        return load_dataset(hf_name, hf_config, split=split)

    manifest = _resolve_local_manifest(_manifest_for_split(cfg, split), config_path, split)
    return load_dataset("json", data_files=str(manifest), split="train")


def split_exists(cfg: dict, config_path: Path, split: str) -> bool:
    try:
        load_split(cfg, config_path, split)
        return True
    except Exception:
        return False


def normalize_batch(batch, text_col: str, text_normalize_fn):
    raw_text = str(batch[text_col] or "")
    batch["raw_text"] = raw_text
    batch["text"] = text_normalize_fn(raw_text)
    return batch


# ---------------------------------------------------------------------------
# Pseudolabel dataset loading
# ---------------------------------------------------------------------------

def _load_filtered_pseudolabels(
    pseudo_json: Path,
    pseudo_score_col: str,
    pseudo_min_score: float,
    pseudo_audio_col: str,
    pseudo_text_col: str,
    num_proc: int,
):
    """Load pseudolabel JSONL, apply score filter, validate columns, add __hf_id stem column."""
    ds = load_dataset("json", data_files=str(pseudo_json), split="train")

    if pseudo_min_score > 0:
        if pseudo_score_col in ds.column_names:
            before = len(ds)
            ds = ds.filter(
                lambda ex: ex[pseudo_score_col] is not None
                and float(ex[pseudo_score_col]) >= pseudo_min_score
            )
            print(
                f"Pseudolabel score filter: kept {len(ds)}/{before} with "
                f"{pseudo_score_col}>={pseudo_min_score}"
            )
        else:
            print(
                f"Warning: score column '{pseudo_score_col}' not found; "
                "skipping pseudolabel min_score filtering."
            )

    if pseudo_audio_col not in ds.column_names:
        raise ValueError(
            f"Pseudolabel audio column '{pseudo_audio_col}' not found in {pseudo_json}. "
            f"Available columns: {ds.column_names}"
        )
    if pseudo_text_col not in ds.column_names:
        raise ValueError(
            f"Text column '{pseudo_text_col}' not found in {pseudo_json}. "
            f"Available columns: {ds.column_names}"
        )

    # Derive HF utterance ID from the audio filename stem.
    ds = ds.map(
        lambda ex: {"__hf_id": Path(str(ex[pseudo_audio_col] or "")).stem},
        num_proc=num_proc,
    )
    return ds


def _join_pseudolabels_to_hf(
    pseudo_ds,
    hf_ds,
    hf_id_col: str,
    text_col: str,
    pseudo_text_col: str,
    pseudo_score_col: str,
) -> any:
    """Match pseudolabel rows to HF audio rows by __hf_id and return a joined dataset."""
    if hf_id_col not in hf_ds.column_names:
        raise ValueError(
            f"HF id column '{hf_id_col}' not found. Available columns: {hf_ds.column_names}"
        )

    hf_ids = [str(x) for x in hf_ds[hf_id_col]]
    id_to_idx: Dict[str, int] = {}
    for i, key in enumerate(hf_ids):
        if key not in id_to_idx:
            id_to_idx[key] = i

    pseudo_ids = [str(x) for x in pseudo_ds["__hf_id"]]
    pseudo_texts = [str(x) for x in pseudo_ds[pseudo_text_col]]
    pseudo_scores = (
        [x for x in pseudo_ds[pseudo_score_col]]
        if pseudo_score_col in pseudo_ds.column_names
        else None
    )

    keep_hf_indices: List[int] = []
    keep_texts: List[str] = []
    keep_scores: Optional[List] = [] if pseudo_scores is not None else None
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
        raise ValueError("Pseudolabel/HF id join produced 0 matches.")

    out_ds = hf_ds.select(keep_hf_indices)
    if text_col in out_ds.column_names:
        out_ds = out_ds.remove_columns(text_col)
    out_ds = out_ds.add_column(text_col, keep_texts)
    if keep_scores is not None and pseudo_score_col not in out_ds.column_names:
        out_ds = out_ds.add_column(pseudo_score_col, keep_scores)

    print(
        f"Pseudolabel join on HF id: matched={len(out_ds)} "
        f"dropped_unmatched={dropped} from pseudo_rows={len(pseudo_ds)}"
    )
    return out_ds


def load_pseudolabel_dataset(cfg: dict, config_path: Path):
    backend = str(get_in(cfg, "dataset.backend", "hf"))
    if backend != "hf":
        raise ValueError(
            "Pseudolabel join currently requires dataset.backend=hf because audio is sourced "
            "from HF splits by id."
        )

    pseudo_json = resolve_path(get_in(cfg, "transcript.jsonl.json_path"), config_path)
    if pseudo_json is None or not pseudo_json.exists():
        raise FileNotFoundError(f"Pseudolabel JSON not found: {pseudo_json}")

    text_col = str(get_in(cfg, "dataset.columns.transcript"))
    pseudo_text_col = str(get_in(cfg, "transcript.jsonl.text_col", text_col))
    audio_col = str(get_in(cfg, "dataset.columns.audio"))
    pseudo_audio_col = str(get_in(cfg, "transcript.jsonl.audio_path_col", "audio_path"))
    pseudo_score_col = str(get_in(cfg, "transcript.jsonl.score_col", "score"))
    pseudo_min_score = float(get_in(cfg, "transcript.jsonl.min_score", 0.0))
    hf_id_col = str(get_in(cfg, "transcript.jsonl.hf_id_col", "id"))
    hf_audio_splits = str(get_in(cfg, "transcript.jsonl.hf_audio_splits", "train,validation"))
    prevent_test_leakage = bool(get_in(cfg, "transcript.jsonl.prevent_test_leakage", True))
    test_split = str(get_in(cfg, "dataset.splits.test", "test"))
    num_proc = int(get_in(cfg, "training.num_proc"))

    pseudo_ds = _load_filtered_pseudolabels(
        pseudo_json, pseudo_score_col, pseudo_min_score,
        pseudo_audio_col, pseudo_text_col, num_proc,
    )

    split_names = [s.strip() for s in hf_audio_splits.split(",") if s.strip()]
    if prevent_test_leakage and test_split in set(split_names):
        raise ValueError(
            f"Pseudolabel HF audio splits include test split '{test_split}' in "
            f"transcript.jsonl.hf_audio_splits={split_names}."
        )

    hf_parts = []
    for split_name in split_names:
        if split_exists(cfg, config_path, split_name):
            hf_parts.append(load_split(cfg, config_path, split_name))
        else:
            print(f"Pseudolabel join: skip missing HF split '{split_name}'")

    if not hf_parts:
        raise ValueError(f"No HF splits available for pseudolabel join: {split_names}")

    hf_ds = hf_parts[0] if len(hf_parts) == 1 else concatenate_datasets(hf_parts)

    if audio_col not in hf_ds.column_names:
        raise ValueError(
            f"HF audio column '{audio_col}' not found. Available columns: {hf_ds.column_names}"
        )

    return _join_pseudolabels_to_hf(
        pseudo_ds, hf_ds, hf_id_col, text_col, pseudo_text_col, pseudo_score_col
    )


# ---------------------------------------------------------------------------
# Training sub-functions
# ---------------------------------------------------------------------------

def _load_datasets(cfg: dict, config_path: Path) -> tuple:
    """Load train, dev, and test datasets according to transcript source config.

    Returns (train_ds, dev_ds, test_ds, source_label).
    """
    transcript_source = str(get_in(cfg, "transcript.source", "inline"))
    transcript_json_type = str(get_in(cfg, "transcript.jsonl.type", "ground_truth"))
    train_split = str(get_in(cfg, "dataset.splits.train", "train"))
    val_split = str(get_in(cfg, "dataset.splits.val", "validation"))
    test_split = str(get_in(cfg, "dataset.splits.test", "test"))
    val_size = float(get_in(cfg, "training.val_size"))
    seed = int(get_in(cfg, "training.seed"))

    source_label = transcript_source
    transcript_map = None

    if transcript_source == "inline":
        train_ds_full = load_split(cfg, config_path, train_split)
    elif transcript_source == "jsonl":
        source_label = f"jsonl:{transcript_json_type}"
        if transcript_json_type == "pseudolabel":
            train_ds_full = load_pseudolabel_dataset(cfg, config_path)
        elif transcript_json_type == "ground_truth":
            transcript_map = load_jsonl_transcript_map(cfg, config_path)
            train_ds_full = apply_jsonl_transcripts(
                load_split(cfg, config_path, train_split), train_split, cfg, transcript_map
            )
        else:
            raise ValueError("transcript.jsonl.type must be one of: ground_truth, pseudolabel")
    else:
        raise ValueError("transcript.source must be one of: inline, jsonl")

    test_ds = load_split(cfg, config_path, test_split)
    if transcript_source == "jsonl" and transcript_json_type == "ground_truth":
        assert transcript_map is not None
        test_ds = apply_jsonl_transcripts(test_ds, test_split, cfg, transcript_map)

    pseudo_dev_json = get_in(cfg, "transcript.jsonl.dev_json_path")
    use_gt_val = transcript_source == "inline" or (
        transcript_source == "jsonl" and transcript_json_type == "ground_truth"
    )
    if use_gt_val and split_exists(cfg, config_path, val_split):
        train_ds = train_ds_full
        dev_ds = load_split(cfg, config_path, val_split)
        if transcript_source == "jsonl" and transcript_json_type == "ground_truth":
            assert transcript_map is not None
            dev_ds = apply_jsonl_transcripts(dev_ds, val_split, cfg, transcript_map)
        print(f"Using existing validation split: {val_split}")
    elif transcript_source == "jsonl" and transcript_json_type == "pseudolabel" and pseudo_dev_json:
        # Reuse pseudolabel loader by temporarily swapping the json_path.
        orig_json = get_in(cfg, "transcript.jsonl.json_path")
        cfg["transcript"]["jsonl"]["json_path"] = pseudo_dev_json
        try:
            dev_ds = load_pseudolabel_dataset(cfg, config_path)
        finally:
            cfg["transcript"]["jsonl"]["json_path"] = orig_json
        train_ds = train_ds_full
        print(f"Using pseudolabel dev JSON: {pseudo_dev_json}")
    else:
        split_ds = train_ds_full.train_test_split(test_size=val_size, seed=seed)
        train_ds = split_ds["train"]
        dev_ds = split_ds["test"]
        print(
            f"No '{val_split}' split found; created train/dev split from "
            f"'{train_split}' with val_size={val_size}"
        )

    return train_ds, dev_ds, test_ds, source_label


def _preprocess_datasets(
    train_ds, dev_ds, test_ds,
    cfg: dict,
    config_path: Path,
    audio_col: str,
    text_col: str,
    test_text_col: str,
    test_audio_col: str,
    text_normalize_fn,
    max_sec: float,
    dry_run: bool,
):
    """Apply audio path prefixing, text normalization, filtering, casting, and column selection."""
    dataset_backend = str(get_in(cfg, "dataset.backend", "hf"))
    local_audio_root = resolve_path(get_in(cfg, "dataset.local.audio_root"), config_path)
    num_proc = int(get_in(cfg, "training.num_proc"))

    if dataset_backend == "local":
        train_ds = maybe_prefix_local_audio_paths(train_ds, audio_col, local_audio_root, num_proc)
        dev_ds = maybe_prefix_local_audio_paths(dev_ds, audio_col, local_audio_root, num_proc)
        test_ds = maybe_prefix_local_audio_paths(test_ds, test_audio_col, local_audio_root, num_proc)

    # Always num_proc=1 for normalizer mapping — the full normalizer is not fork-safe.
    train_ds = train_ds.map(lambda b: normalize_batch(b, text_col, text_normalize_fn), num_proc=1)
    dev_ds = dev_ds.map(lambda b: normalize_batch(b, text_col, text_normalize_fn), num_proc=1)
    test_ds = test_ds.map(lambda b: normalize_batch(b, test_text_col, text_normalize_fn), num_proc=1)

    train_ds = train_ds.filter(lambda x: len(x["text"]) > 0)
    dev_ds = dev_ds.filter(lambda x: len(x["text"]) > 0)
    test_ds = test_ds.filter(lambda x: len(x["text"]) > 0)

    if audio_col not in test_ds.column_names and test_audio_col in test_ds.column_names:
        test_ds = test_ds.rename_column(test_audio_col, audio_col)

    if audio_col not in train_ds.column_names or audio_col not in dev_ds.column_names:
        raise ValueError(
            f"audio column '{audio_col}' not found in training/dev datasets. "
            f"Available train columns: {train_ds.column_names}; dev columns: {dev_ds.column_names}"
        )
    if audio_col not in test_ds.column_names:
        raise ValueError(
            f"audio column '{audio_col}' not found in test dataset. "
            f"Consider setting dataset.columns.test_audio (current: {test_audio_col})."
        )

    train_ds = train_ds.cast_column(audio_col, Audio(sampling_rate=TARGET_SR))
    dev_ds = dev_ds.cast_column(audio_col, Audio(sampling_rate=TARGET_SR))
    test_ds = test_ds.cast_column(audio_col, Audio(sampling_rate=TARGET_SR))

    if max_sec > 0 and not dry_run:
        train_ds = train_ds.filter(lambda ex: keep_max_duration(ex, audio_col, max_sec), num_proc=num_proc)
        dev_ds = dev_ds.filter(lambda ex: keep_max_duration(ex, audio_col, max_sec), num_proc=num_proc)
        test_ds = test_ds.filter(lambda ex: keep_max_duration(ex, audio_col, max_sec), num_proc=num_proc)

    def _cols(ds, base):
        return base + (["__utt_id"] if "__utt_id" in ds.column_names else [])

    train_ds = train_ds.select_columns(_cols(train_ds, [audio_col, "text"]))
    dev_ds = dev_ds.select_columns(_cols(dev_ds, [audio_col, "text"]))
    test_ds = test_ds.select_columns(_cols(test_ds, [audio_col, "text", "raw_text"]))

    return train_ds, dev_ds, test_ds


def _build_training_args(cfg: dict, out_dir: Path):
    from transformers import TrainingArguments

    return TrainingArguments(
        output_dir=str(out_dir),
        fp16=True,
        per_device_train_batch_size=int(get_in(cfg, "training.bs")),
        per_device_eval_batch_size=int(get_in(cfg, "training.bs")),
        gradient_accumulation_steps=int(get_in(cfg, "training.grad_accum")),
        learning_rate=float(get_in(cfg, "training.lr")),
        lr_scheduler_type=str(get_in(cfg, "training.lr_scheduler")),
        warmup_ratio=float(get_in(cfg, "training.warmup_ratio")),
        weight_decay=float(get_in(cfg, "training.weight_decay")),
        num_train_epochs=float(get_in(cfg, "training.epochs")),
        group_by_length=False,
        eval_strategy="steps",
        save_strategy="steps",
        eval_steps=int(get_in(cfg, "training.eval_steps")),
        save_steps=int(get_in(cfg, "training.save_steps")),
        load_best_model_at_end=True,
        metric_for_best_model=str(get_in(cfg, "training.best_metric")),
        greater_is_better=False,
        logging_steps=int(get_in(cfg, "training.logging_steps")),
        save_total_limit=int(get_in(cfg, "training.save_total_limit")),
        max_grad_norm=float(get_in(cfg, "training.max_grad_norm")),
        report_to="none",
        remove_unused_columns=False,
    )


def _evaluate_test_split(trainer, processor, test_ds, text_normalize_fn, wer_metric, out_dir: Path) -> dict:
    print("\nEvaluating on TEST split...")
    pred_out = trainer.predict(test_ds, metric_key_prefix="test")
    metrics = dict(pred_out.metrics)

    pred_ids = np.argmax(pred_out.predictions, axis=-1)
    pred_text = processor.batch_decode(pred_ids)
    test_refs_raw = [str(x) for x in test_ds["raw_text"]]
    test_refs_norm = [str(x) for x in test_ds["text"]]
    pred_text_norm = [text_normalize_fn(s) for s in pred_text]

    metrics["test_wer_raw_ref"] = wer_metric.compute(predictions=pred_text, references=test_refs_raw)
    metrics["test_wer_norm_ref"] = wer_metric.compute(predictions=pred_text_norm, references=test_refs_norm)

    print("TEST metrics:", metrics)
    metrics_path = out_dir / "test_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Wrote TEST metrics -> {metrics_path}")
    return metrics


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run_training(cfg: dict, config_path: Path, dry_run: bool = False):
    vocab_path = resolve_path(get_in(cfg, "vocab.out_path"), config_path)
    if vocab_path is None or not vocab_path.exists():
        raise FileNotFoundError(
            f"Vocab file not found: {vocab_path}. Run ctc-build-vocab first or adjust vocab.out_path."
        )

    out_dir = resolve_path(get_in(cfg, "training.out_dir"), config_path)
    if out_dir is None:
        raise ValueError("training.out_dir is required")
    out_dir.mkdir(parents=True, exist_ok=True)
    save_yaml(out_dir / "resolved_config.yaml", cfg)

    model_name_or_path = str(get_in(cfg, "model.name_or_path"))
    audio_col = str(get_in(cfg, "dataset.columns.audio"))
    text_col = str(get_in(cfg, "dataset.columns.transcript"))
    test_audio_col = str(get_in(cfg, "dataset.columns.test_audio", audio_col))
    test_text_col = get_in(cfg, "dataset.columns.test_transcript") or text_col
    max_sec = float(get_in(cfg, "training.max_sec"))
    normalizer_yaml = get_in(cfg, "normalization.normalizer_yaml")

    text_normalize_fn, _, used_yaml = build_normalizer_fn(normalizer_yaml, config_path)

    train_ds, dev_ds, test_ds, source_label = _load_datasets(cfg, config_path)

    train_ds, dev_ds, test_ds = _preprocess_datasets(
        train_ds, dev_ds, test_ds,
        cfg, config_path,
        audio_col, text_col, test_text_col, test_audio_col,
        text_normalize_fn, max_sec, dry_run,
    )

    summary = {
        "model_name_or_path": model_name_or_path,
        "dataset_backend": str(get_in(cfg, "dataset.backend", "hf")),
        "transcript_source": source_label,
        "normalizer_yaml": used_yaml,
        "splits": {
            "train": len(train_ds),
            "dev": len(dev_ds),
            "test": len(test_ds),
        },
        "columns": {
            "audio": audio_col,
            "transcript": text_col,
            "test_transcript": test_text_col,
        },
    }
    (out_dir / "dataset_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    if dry_run:
        print("Dry run: resolved training summary")
        print(json.dumps(summary, indent=2))
        return summary

    from transformers import Trainer, Wav2Vec2ForCTC

    processor = build_processor(vocab_path)
    data_collator = DataCollatorCTCWithPadding(processor=processor, audio_col=audio_col)
    wer = evaluate.load("wer")

    def compute_metrics(pred):
        pred_ids = np.argmax(pred.predictions, axis=-1)
        label_ids = pred.label_ids.copy()
        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
        pred_str = processor.batch_decode(pred_ids)
        label_str = processor.batch_decode(label_ids, group_tokens=False)
        pred_norm = [text_normalize_fn(s) for s in pred_str]
        label_norm = [text_normalize_fn(s) for s in label_str]
        return {
            "wer_decoded_label_ref": wer.compute(predictions=pred_str, references=label_str),
            "wer_decoded_norm": wer.compute(predictions=pred_norm, references=label_norm),
        }

    model = Wav2Vec2ForCTC.from_pretrained(
        model_name_or_path,
        pad_token_id=processor.tokenizer.pad_token_id,
        vocab_size=len(processor.tokenizer),
        ctc_loss_reduction="mean",
    )
    for p in model.wav2vec2.parameters():
        p.requires_grad = True
    for p in model.lm_head.parameters():
        p.requires_grad = True

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(
        f"Source={source_label}; Splits: train={len(train_ds)} "
        f"eval(dev)={len(dev_ds)} test={len(test_ds)}; trainable_params={trainable}"
    )

    trainer = Trainer(
        model=model,
        args=_build_training_args(cfg, out_dir),
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        train_dataset=train_ds,
        eval_dataset=dev_ds,
    )

    trainer.train()
    trainer.save_model(str(out_dir))
    processor.save_pretrained(str(out_dir))

    return _evaluate_test_split(trainer, processor, test_ds, text_normalize_fn, wer, out_dir)
