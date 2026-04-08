"""Tests for ctc_framework.pipelines.vocab_pipeline."""

import json
import pytest

from ctc_framework.config.loader import load_yaml
from ctc_framework.pipelines.vocab_pipeline import (
    _apply_score_filter,
    _collect_vocab,
    run_vocab_build,
)


# ---------------------------------------------------------------------------
# _apply_score_filter
# ---------------------------------------------------------------------------

def test_apply_score_filter_keeps_above_threshold():
    from datasets import Dataset
    ds = Dataset.from_dict({"text": ["a", "b", "c"], "score": [0.9, 0.5, 0.7]})
    out = _apply_score_filter(ds, "score", 0.6, "test")
    assert len(out) == 2
    assert list(out["score"]) == [0.9, 0.7]


def test_apply_score_filter_zero_min_score_skips():
    from datasets import Dataset
    ds = Dataset.from_dict({"text": ["a", "b"], "score": [0.1, 0.9]})
    out = _apply_score_filter(ds, "score", 0.0, "test")
    assert len(out) == 2  # unchanged


def test_apply_score_filter_missing_column_is_noop():
    from datasets import Dataset
    ds = Dataset.from_dict({"text": ["a", "b"]})
    out = _apply_score_filter(ds, "score", 0.6, "test")
    assert len(out) == 2  # no column → warning, no filter


# ---------------------------------------------------------------------------
# _collect_vocab
# ---------------------------------------------------------------------------

def test_collect_vocab_basic(tmp_path):
    from datasets import Dataset
    from ctc_framework.pipelines.utils import build_normalizer_fn

    ds = Dataset.from_dict({"text": ["ab", "bc"]})
    text_norm_fn, _, _ = build_normalizer_fn(None, tmp_path / "cfg.yaml")
    vocab, empty, size = _collect_vocab(ds, "text", "test", text_norm_fn)
    assert "a" in vocab
    assert "b" in vocab
    assert "c" in vocab
    assert empty == 0
    assert size == 2


def test_collect_vocab_empty_after_norm_counted(tmp_path):
    from datasets import Dataset
    from ctc_framework.pipelines.utils import build_normalizer_fn

    # Punctuation-only string becomes empty after normalization
    ds = Dataset.from_dict({"text": ["!!!", "hello"]})
    text_norm_fn, _, _ = build_normalizer_fn(None, tmp_path / "cfg.yaml")
    vocab, empty, size = _collect_vocab(ds, "text", "test", text_norm_fn)
    assert empty == 1
    assert size == 2


def test_collect_vocab_missing_column_raises(tmp_path):
    from datasets import Dataset
    from ctc_framework.pipelines.utils import build_normalizer_fn

    ds = Dataset.from_dict({"wrong_col": ["hello"]})
    text_norm_fn, _, _ = build_normalizer_fn(None, tmp_path / "cfg.yaml")
    with pytest.raises(ValueError, match="Text column"):
        _collect_vocab(ds, "text", "test", text_norm_fn)


# ---------------------------------------------------------------------------
# run_vocab_build (dry-run, no file I/O)
# ---------------------------------------------------------------------------

def test_run_vocab_build_dry_run_returns_summary(local_config_path, local_cfg):
    summary = run_vocab_build(local_cfg, local_config_path, dry_run=True)
    assert "vocab_size" in summary
    assert "source" in summary
    assert "normalizer_yaml" in summary
    assert summary["vocab_size"] > 3  # at least a-z characters present


def test_run_vocab_build_dry_run_no_files_written(local_config_path, local_cfg, tmp_path):
    run_vocab_build(local_cfg, local_config_path, dry_run=True)
    vocab_dir = tmp_path / "vocab"
    assert not vocab_dir.exists()


def test_run_vocab_build_writes_vocab_and_summary(local_config_path, local_cfg, tmp_path):
    run_vocab_build(local_cfg, local_config_path, dry_run=False)

    vocab_path = tmp_path / "vocab" / "vocab.json"
    assert vocab_path.exists()
    vocab = json.loads(vocab_path.read_text())
    assert "|" in vocab
    assert "[UNK]" in vocab
    assert "[PAD]" in vocab

    summary_path = tmp_path / "vocab" / "vocab_summary.json"
    assert summary_path.exists()
    summary = json.loads(summary_path.read_text())
    assert summary["vocab_size"] == len(vocab)


def test_run_vocab_build_special_tokens_at_end(local_config_path, local_cfg, tmp_path):
    """Special tokens must have the highest indices (they come last)."""
    run_vocab_build(local_cfg, local_config_path, dry_run=False)
    vocab_path = tmp_path / "vocab" / "vocab.json"
    vocab = json.loads(vocab_path.read_text())
    max_idx = max(vocab.values())
    assert vocab["[PAD]"] == max_idx
    assert vocab["[UNK]"] == max_idx - 1
    assert vocab["|"] == max_idx - 2


def test_run_vocab_build_missing_out_path_raises(local_config_path):
    from ctc_framework.config.loader import load_yaml, apply_overrides
    cfg = apply_overrides(load_yaml(local_config_path), ["vocab.out_path=null"])
    with pytest.raises(ValueError, match="vocab.out_path"):
        run_vocab_build(cfg, local_config_path, dry_run=True)


def test_run_vocab_build_with_score_filter(tmp_path, local_dataset):
    """Vocab built from a scored pseudolabel JSONL with min_score filter."""
    rows = [
        {"id": "a", "text": "hello world", "score": 0.9},
        {"id": "b", "text": "zzzzz", "score": 0.2},  # below threshold, chars excluded
    ]
    jsonl = tmp_path / "pseudo.jsonl"
    jsonl.write_text("\n".join(__import__("json").dumps(r) for r in rows), encoding="utf-8")

    cfg_text = f"""\
dataset:
  backend: local
  local:
    manifests:
      train: {local_dataset["train_manifest"]}
  columns:
    transcript: text

transcript:
  source: jsonl
  jsonl:
    type: ground_truth
    json_path: {jsonl}
    text_col: text
  join:
    json_key: id
    dataset_key: id

normalization:
  normalizer_yaml: null

vocab:
  mode: dataset_only
  out_path: {tmp_path / "vocab" / "vocab.json"}
  score_col: score
  min_score: 0.6
"""
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(cfg_text, encoding="utf-8")
    from ctc_framework.config.loader import load_yaml
    cfg = load_yaml(cfg_path)
    summary = run_vocab_build(cfg, cfg_path, dry_run=True)
    assert summary["vocab_size"] > 0
