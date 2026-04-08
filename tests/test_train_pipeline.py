"""Tests for ctc_framework.pipelines.train_pipeline (dry-run only — no model loading).

All tests use local JSONL manifests and WAV fixtures from conftest.py.
No HuggingFace downloads or GPU usage.
"""

import json
import pytest

from ctc_framework.config.loader import apply_overrides, load_yaml
from ctc_framework.pipelines.train_pipeline import (
    _join_pseudolabels_to_hf,
    _load_datasets,
    _load_filtered_pseudolabels,
    run_training,
)


# ---------------------------------------------------------------------------
# run_training dry-run — inline transcript mode
# ---------------------------------------------------------------------------

def test_run_training_dry_run_inline_returns_summary(local_config_path, local_cfg, tmp_path):
    # Need a vocab file to pass the pre-flight check
    vocab_path = tmp_path / "vocab" / "vocab.json"
    vocab_path.parent.mkdir(parents=True)
    vocab_path.write_text('{"a": 0, "|": 1, "[UNK]": 2, "[PAD]": 3}', encoding="utf-8")
    cfg = apply_overrides(local_cfg, [f"vocab.out_path={vocab_path}"])

    summary = run_training(cfg, local_config_path, dry_run=True)

    assert "splits" in summary
    assert summary["splits"]["train"] >= 1
    assert summary["splits"]["test"] >= 1
    assert summary["transcript_source"] == "inline"
    assert summary["normalizer_yaml"] is not None


def test_run_training_dry_run_writes_resolved_config(local_config_path, local_cfg, tmp_path):
    vocab_path = tmp_path / "vocab" / "vocab.json"
    vocab_path.parent.mkdir(parents=True)
    vocab_path.write_text('{"a": 0, "|": 1, "[UNK]": 2, "[PAD]": 3}', encoding="utf-8")
    cfg = apply_overrides(local_cfg, [f"vocab.out_path={vocab_path}"])

    run_training(cfg, local_config_path, dry_run=True)

    out_dir = tmp_path / "runs" / "test_run"
    assert (out_dir / "resolved_config.yaml").exists()


def test_run_training_dry_run_writes_dataset_summary(local_config_path, local_cfg, tmp_path):
    vocab_path = tmp_path / "vocab" / "vocab.json"
    vocab_path.parent.mkdir(parents=True)
    vocab_path.write_text('{"a": 0, "|": 1, "[UNK]": 2, "[PAD]": 3}', encoding="utf-8")
    cfg = apply_overrides(local_cfg, [f"vocab.out_path={vocab_path}"])

    run_training(cfg, local_config_path, dry_run=True)

    summary_path = tmp_path / "runs" / "test_run" / "dataset_summary.json"
    assert summary_path.exists()
    summary = json.loads(summary_path.read_text())
    assert "splits" in summary
    assert "columns" in summary


def test_run_training_dry_run_missing_vocab_raises(local_config_path, local_cfg, tmp_path):
    cfg = apply_overrides(local_cfg, ["vocab.out_path=/nonexistent/vocab.json"])
    with pytest.raises(FileNotFoundError, match="Vocab file not found"):
        run_training(cfg, local_config_path, dry_run=True)


# ---------------------------------------------------------------------------
# run_training dry-run — jsonl ground_truth transcript mode
# ---------------------------------------------------------------------------

def test_run_training_dry_run_jsonl_gt(local_config_path, local_cfg, tmp_path, transcript_jsonl):
    vocab_path = tmp_path / "vocab" / "vocab.json"
    vocab_path.parent.mkdir(parents=True)
    vocab_path.write_text('{"a": 0, "|": 1, "[UNK]": 2, "[PAD]": 3}', encoding="utf-8")

    cfg = apply_overrides(local_cfg, [
        f"vocab.out_path={vocab_path}",
        "transcript.source=jsonl",
        "transcript.jsonl.type=ground_truth",
        f"transcript.jsonl.json_path={transcript_jsonl}",
        "transcript.join.dataset_key=id",
        "transcript.join.json_key=id",
        "transcript.join.strict=false",
    ])

    summary = run_training(cfg, local_config_path, dry_run=True)
    assert summary["transcript_source"] == "jsonl:ground_truth"
    assert summary["splits"]["train"] >= 1


# ---------------------------------------------------------------------------
# _load_filtered_pseudolabels
# ---------------------------------------------------------------------------

def test_load_filtered_pseudolabels_basic(tmp_path, pseudolabel_jsonl):
    ds = _load_filtered_pseudolabels(
        pseudo_json=pseudolabel_jsonl,
        pseudo_score_col="score",
        pseudo_min_score=0.0,
        pseudo_audio_col="audio_path",
        pseudo_text_col="text",
        num_proc=1,
    )
    assert len(ds) == 3
    assert "__hf_id" in ds.column_names


def test_load_filtered_pseudolabels_score_filter(tmp_path, local_dataset):
    import json as _json
    rows = [
        {"audio_path": "a.wav", "text": "hello", "score": 0.9},
        {"audio_path": "b.wav", "text": "world", "score": 0.3},
    ]
    jsonl = tmp_path / "pseudo.jsonl"
    jsonl.write_text("\n".join(_json.dumps(r) for r in rows), encoding="utf-8")

    ds = _load_filtered_pseudolabels(
        pseudo_json=jsonl,
        pseudo_score_col="score",
        pseudo_min_score=0.6,
        pseudo_audio_col="audio_path",
        pseudo_text_col="text",
        num_proc=1,
    )
    assert len(ds) == 1
    assert ds["text"][0] == "hello"


def test_load_filtered_pseudolabels_hf_id_is_stem(tmp_path):
    import json as _json
    rows = [{"audio_path": "/some/path/utt-001.wav", "text": "hello", "score": 1.0}]
    jsonl = tmp_path / "pseudo.jsonl"
    jsonl.write_text(_json.dumps(rows[0]), encoding="utf-8")

    ds = _load_filtered_pseudolabels(
        pseudo_json=jsonl,
        pseudo_score_col="score",
        pseudo_min_score=0.0,
        pseudo_audio_col="audio_path",
        pseudo_text_col="text",
        num_proc=1,
    )
    assert ds["__hf_id"][0] == "utt-001"


# ---------------------------------------------------------------------------
# _join_pseudolabels_to_hf
# ---------------------------------------------------------------------------

def test_join_pseudolabels_to_hf_basic():
    from datasets import Dataset

    pseudo_ds = Dataset.from_dict({
        "__hf_id": ["utt-001", "utt-002"],
        "text": ["hello", "world"],
        "score": [0.9, 0.8],
    })
    hf_ds = Dataset.from_dict({
        "id": ["utt-001", "utt-002", "utt-003"],
        "audio": ["a.wav", "b.wav", "c.wav"],
    })

    out = _join_pseudolabels_to_hf(
        pseudo_ds, hf_ds,
        hf_id_col="id",
        text_col="text",
        pseudo_text_col="text",
        pseudo_score_col="score",
    )
    assert len(out) == 2
    assert list(out["text"]) == ["hello", "world"]
    assert "audio" in out.column_names


def test_join_pseudolabels_to_hf_zero_matches_raises():
    from datasets import Dataset

    pseudo_ds = Dataset.from_dict({
        "__hf_id": ["x", "y"],
        "text": ["a", "b"],
        "score": [0.9, 0.8],
    })
    hf_ds = Dataset.from_dict({
        "id": ["utt-001"],
        "audio": ["a.wav"],
    })

    with pytest.raises(ValueError, match="0 matches"):
        _join_pseudolabels_to_hf(
            pseudo_ds, hf_ds,
            hf_id_col="id",
            text_col="text",
            pseudo_text_col="text",
            pseudo_score_col="score",
        )


def test_join_pseudolabels_to_hf_drops_unmatched():
    from datasets import Dataset

    pseudo_ds = Dataset.from_dict({
        "__hf_id": ["utt-001", "missing"],
        "text": ["hello", "dropped"],
        "score": [0.9, 0.9],
    })
    hf_ds = Dataset.from_dict({
        "id": ["utt-001"],
        "audio": ["a.wav"],
    })

    out = _join_pseudolabels_to_hf(
        pseudo_ds, hf_ds,
        hf_id_col="id",
        text_col="text",
        pseudo_text_col="text",
        pseudo_score_col="score",
    )
    assert len(out) == 1
    assert out["text"][0] == "hello"


# ---------------------------------------------------------------------------
# _load_datasets — split routing
# ---------------------------------------------------------------------------

def test_load_datasets_inline_returns_three_splits(local_config_path, local_cfg):
    train_ds, dev_ds, test_ds, label = _load_datasets(local_cfg, local_config_path)
    assert label == "inline"
    assert len(train_ds) >= 1
    assert len(dev_ds) >= 1
    assert len(test_ds) >= 1


def test_load_datasets_invalid_source_raises(local_config_path, local_cfg):
    cfg = apply_overrides(local_cfg, ["transcript.source=unknown"])
    with pytest.raises(ValueError, match="transcript.source"):
        _load_datasets(cfg, local_config_path)
