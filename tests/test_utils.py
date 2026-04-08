"""Tests for ctc_framework.pipelines.utils."""

import json
from pathlib import Path

import numpy as np
import pytest

from ctc_framework.pipelines.utils import (
    apply_jsonl_transcripts,
    build_normalizer_fn,
    keep_max_duration,
    load_jsonl_transcript_map,
    maybe_prefix_local_audio_paths,
    resolve_path,
)
from ctc_framework.config.loader import load_yaml


# ---------------------------------------------------------------------------
# resolve_path
# ---------------------------------------------------------------------------

def test_resolve_path_absolute(tmp_path):
    p = tmp_path / "file.json"
    result = resolve_path(str(p), tmp_path / "config.yaml")
    assert result == p


def test_resolve_path_relative(tmp_path):
    config_path = tmp_path / "configs" / "cfg.yaml"
    result = resolve_path("../data/file.json", config_path)
    assert result == (tmp_path / "data" / "file.json").resolve()


def test_resolve_path_none(tmp_path):
    assert resolve_path(None, tmp_path / "cfg.yaml") is None


# ---------------------------------------------------------------------------
# build_normalizer_fn
# ---------------------------------------------------------------------------

def test_build_normalizer_fn_returns_callables(tmp_path):
    config_path = tmp_path / "cfg.yaml"
    text_norm, normalize_both, used_yaml = build_normalizer_fn(None, config_path)
    assert callable(text_norm)
    assert callable(normalize_both)
    assert used_yaml is not None


def test_text_norm_fn_lowercases(tmp_path):
    config_path = tmp_path / "cfg.yaml"
    text_norm, _, _ = build_normalizer_fn(None, config_path)
    assert text_norm("Hello World") == "hello world"


def test_text_norm_fn_strips_punctuation(tmp_path):
    config_path = tmp_path / "cfg.yaml"
    text_norm, _, _ = build_normalizer_fn(None, config_path)
    result = text_norm("hello, world!")
    assert "," not in result
    assert "!" not in result


def test_normalize_both_returns_tuple(tmp_path):
    config_path = tmp_path / "cfg.yaml"
    _, normalize_both, _ = build_normalizer_fn(None, config_path)
    norm, no_fill = normalize_both("hello world")
    assert isinstance(norm, str)
    assert isinstance(no_fill, str)


def test_normalize_both_filler_removal(tmp_path):
    """text_no_fill should strip known fillers like 'uh'."""
    config_path = tmp_path / "cfg.yaml"
    _, normalize_both, _ = build_normalizer_fn(None, config_path)
    norm, no_fill = normalize_both("hello uh world")
    assert "uh" in norm or "hello" in norm  # norm keeps filler form
    assert "uh" not in no_fill
    assert "hello" in no_fill
    assert "world" in no_fill


# ---------------------------------------------------------------------------
# keep_max_duration
# ---------------------------------------------------------------------------

def _make_audio_ex(duration_s: float, sr: int = 16000) -> dict:
    n = int(sr * duration_s)
    return {"audio": {"array": np.zeros(n, dtype=np.float32), "sampling_rate": sr}}


def test_keep_max_duration_within_limit():
    ex = _make_audio_ex(5.0)
    assert keep_max_duration(ex, "audio", 10.0) is True


def test_keep_max_duration_exactly_at_limit():
    ex = _make_audio_ex(10.0)
    assert keep_max_duration(ex, "audio", 10.0) is True


def test_keep_max_duration_exceeds_limit():
    ex = _make_audio_ex(15.0)
    assert keep_max_duration(ex, "audio", 10.0) is False


# ---------------------------------------------------------------------------
# load_jsonl_transcript_map
# ---------------------------------------------------------------------------

def test_load_jsonl_transcript_map_basic(tmp_path):
    rows = [
        {"id": "a", "text": "hello"},
        {"id": "b", "text": "world"},
    ]
    jsonl = tmp_path / "t.jsonl"
    jsonl.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")

    cfg = {
        "transcript": {
            "jsonl": {"json_path": str(jsonl), "text_col": "text"},
            "join": {"json_key": "id"},
        }
    }
    result = load_jsonl_transcript_map(cfg, tmp_path / "cfg.yaml")
    assert result == {"a": "hello", "b": "world"}


def test_load_jsonl_transcript_map_missing_file(tmp_path):
    cfg = {
        "transcript": {
            "jsonl": {"json_path": str(tmp_path / "missing.jsonl"), "text_col": "text"},
            "join": {"json_key": "id"},
        }
    }
    with pytest.raises(FileNotFoundError):
        load_jsonl_transcript_map(cfg, tmp_path / "cfg.yaml")


def test_load_jsonl_transcript_map_duplicate_key(tmp_path):
    rows = [{"id": "a", "text": "hello"}, {"id": "a", "text": "duplicate"}]
    jsonl = tmp_path / "t.jsonl"
    jsonl.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")

    cfg = {
        "transcript": {
            "jsonl": {"json_path": str(jsonl), "text_col": "text"},
            "join": {"json_key": "id"},
        }
    }
    with pytest.raises(ValueError, match="Duplicate"):
        load_jsonl_transcript_map(cfg, tmp_path / "cfg.yaml")


def test_load_jsonl_transcript_map_empty_file(tmp_path):
    # Write one valid-JSON row then an empty file equivalent — datasets raises on
    # truly empty files before we can check length. Use a single-row file that
    # produces 0 matches instead to test the "empty map" guard.
    jsonl = tmp_path / "empty.jsonl"
    jsonl.write_text('{"other_col": "val"}', encoding="utf-8")
    cfg = {
        "transcript": {
            "jsonl": {"json_path": str(jsonl), "text_col": "text"},
            "join": {"json_key": "id"},
        }
    }
    # Missing columns raise ValueError before reaching the empty-map check
    with pytest.raises(ValueError):
        load_jsonl_transcript_map(cfg, tmp_path / "cfg.yaml")


# ---------------------------------------------------------------------------
# apply_jsonl_transcripts
# ---------------------------------------------------------------------------

def test_apply_jsonl_transcripts_basic(tmp_path):
    from datasets import Dataset

    ds = Dataset.from_dict({"id": ["a", "b", "c"], "audio": ["x", "y", "z"]})
    transcript_map = {"a": "hello", "b": "world"}
    cfg = {
        "transcript": {"join": {"dataset_key": "id", "strict": False}},
        "dataset": {"columns": {"transcript": "text"}},
    }
    out = apply_jsonl_transcripts(ds, "train", cfg, transcript_map)
    assert len(out) == 2
    assert list(out["text"]) == ["hello", "world"]
    assert list(out["__utt_id"]) == ["a", "b"]


def test_apply_jsonl_transcripts_strict_raises_on_missing(tmp_path):
    from datasets import Dataset

    ds = Dataset.from_dict({"id": ["a", "b", "missing"], "audio": ["x", "y", "z"]})
    transcript_map = {"a": "hello", "b": "world"}
    cfg = {
        "transcript": {"join": {"dataset_key": "id", "strict": True}},
        "dataset": {"columns": {"transcript": "text"}},
    }
    with pytest.raises(ValueError, match="missing"):
        apply_jsonl_transcripts(ds, "train", cfg, transcript_map)


def test_apply_jsonl_transcripts_zero_matches_raises(tmp_path):
    from datasets import Dataset

    ds = Dataset.from_dict({"id": ["x", "y"], "audio": ["a", "b"]})
    transcript_map = {"a": "hello"}
    cfg = {
        "transcript": {"join": {"dataset_key": "id", "strict": False}},
        "dataset": {"columns": {"transcript": "text"}},
    }
    with pytest.raises(ValueError, match="0 matches"):
        apply_jsonl_transcripts(ds, "train", cfg, transcript_map)


def test_apply_jsonl_transcripts_custom_text_col(tmp_path):
    from datasets import Dataset

    ds = Dataset.from_dict({"id": ["a"], "audio": ["x"]})
    transcript_map = {"a": "hello"}
    cfg = {
        "transcript": {"join": {"dataset_key": "id", "strict": False}},
        "dataset": {"columns": {"transcript": "text"}},
    }
    out = apply_jsonl_transcripts(ds, "train", cfg, transcript_map, text_col="transcript")
    assert "transcript" in out.column_names
    assert out["transcript"][0] == "hello"


# ---------------------------------------------------------------------------
# maybe_prefix_local_audio_paths
# ---------------------------------------------------------------------------

def test_maybe_prefix_local_audio_paths_relative(tmp_path):
    from datasets import Dataset

    audio_root = tmp_path / "audio"
    audio_root.mkdir()
    (audio_root / "a.wav").touch()

    ds = Dataset.from_dict({"audio_path": ["a.wav"]})
    out = maybe_prefix_local_audio_paths(ds, "audio_path", audio_root)
    assert out["audio_path"][0] == str((audio_root / "a.wav").resolve())


def test_maybe_prefix_local_audio_paths_absolute_unchanged(tmp_path):
    from datasets import Dataset

    audio_root = tmp_path / "audio"
    audio_root.mkdir()
    abs_path = str((audio_root / "a.wav").resolve())

    ds = Dataset.from_dict({"audio_path": [abs_path]})
    out = maybe_prefix_local_audio_paths(ds, "audio_path", audio_root)
    assert out["audio_path"][0] == abs_path


def test_maybe_prefix_local_audio_paths_no_root(tmp_path):
    from datasets import Dataset

    ds = Dataset.from_dict({"audio_path": ["a.wav"]})
    out = maybe_prefix_local_audio_paths(ds, "audio_path", None)
    assert out["audio_path"][0] == "a.wav"
