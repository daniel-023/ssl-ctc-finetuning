"""Shared fixtures for ssl-ctc-finetuning tests.

All fixtures are local-only — no HF downloads, no GPU, no model loading.
Audio fixtures produce minimal valid WAV files (16-bit PCM, 16 kHz, 0.5 s).
"""

import json
import struct
import wave
from pathlib import Path

import numpy as np
import pytest

from ctc_framework.config.loader import apply_overrides, load_yaml


# ---------------------------------------------------------------------------
# WAV helpers
# ---------------------------------------------------------------------------

def _write_wav(path: Path, duration_s: float = 0.5, sr: int = 16000) -> Path:
    """Write a silent mono WAV file at 16 kHz."""
    n_samples = int(sr * duration_s)
    samples = (np.zeros(n_samples, dtype=np.int16)).tobytes()
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(sr)
        wf.writeframes(samples)
    return path


# ---------------------------------------------------------------------------
# JSONL helpers
# ---------------------------------------------------------------------------

def _write_jsonl(path: Path, rows: list) -> Path:
    path.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# Local dataset fixture
# ---------------------------------------------------------------------------

@pytest.fixture()
def local_dataset(tmp_path):
    """A minimal local dataset: 3 audio files + 3-split JSONL manifests.

    Returns a dict with keys:
        audio_dir, train_manifest, val_manifest, test_manifest, audio_paths
    """
    audio_dir = tmp_path / "audio"
    audio_dir.mkdir()

    utterances = [
        {"id": "utt-001", "text": "hello world"},
        {"id": "utt-002", "text": "this is a test"},
        {"id": "utt-003", "text": "the quick brown fox"},
    ]
    audio_paths = {}
    for utt in utterances:
        wav = audio_dir / f"{utt['id']}.wav"
        _write_wav(wav)
        audio_paths[utt["id"]] = str(wav)

    def _manifest_rows(utts):
        return [
            {"id": u["id"], "audio_path": audio_paths[u["id"]], "text": u["text"]}
            for u in utts
        ]

    manifests_dir = tmp_path / "manifests"
    manifests_dir.mkdir()
    train_manifest = _write_jsonl(manifests_dir / "train.jsonl", _manifest_rows(utterances[:2]))
    val_manifest = _write_jsonl(manifests_dir / "val.jsonl", _manifest_rows(utterances[1:2]))
    test_manifest = _write_jsonl(manifests_dir / "test.jsonl", _manifest_rows(utterances[2:]))

    return {
        "audio_dir": audio_dir,
        "train_manifest": train_manifest,
        "val_manifest": val_manifest,
        "test_manifest": test_manifest,
        "audio_paths": audio_paths,
        "utterances": utterances,
    }


@pytest.fixture()
def transcript_jsonl(tmp_path, local_dataset):
    """External ground-truth JSONL keyed by utterance ID."""
    rows = [
        {"id": u["id"], "text": u["text"]}
        for u in local_dataset["utterances"]
    ]
    path = tmp_path / "transcripts.jsonl"
    _write_jsonl(path, rows)
    return path


@pytest.fixture()
def pseudolabel_jsonl(tmp_path, local_dataset):
    """Pseudolabel JSONL with audio_path and score fields."""
    rows = [
        {
            "audio_path": local_dataset["audio_paths"][u["id"]],
            "text": u["text"],
            "score": 0.85,
        }
        for u in local_dataset["utterances"]
    ]
    path = tmp_path / "pseudolabels.jsonl"
    _write_jsonl(path, rows)
    return path


# ---------------------------------------------------------------------------
# Config fixture factory
# ---------------------------------------------------------------------------

@pytest.fixture()
def local_config_path(tmp_path, local_dataset):
    """Write a minimal local-backend YAML config to tmp_path and return its path."""
    cfg_text = f"""\
dataset:
  backend: local
  splits:
    train: train
    val: validation
    test: test
  columns:
    audio: audio_path
    transcript: text
    test_audio: audio_path
    id: id
  local:
    manifests:
      train: {local_dataset["train_manifest"]}
      val: {local_dataset["val_manifest"]}
      test: {local_dataset["test_manifest"]}
    audio_root: null

transcript:
  source: inline

normalization:
  normalizer_yaml: null

vocab:
  mode: dataset_only
  out_path: {tmp_path / "vocab" / "vocab.json"}

training:
  out_dir: {tmp_path / "runs" / "test_run"}
  max_sec: 30.0
  num_proc: 1
  seed: 42
  val_size: 0.5
  bs: 2
  grad_accum: 1
  lr: 1.0e-4
  epochs: 1.0
  lr_scheduler: cosine
  warmup_ratio: 0.0
  weight_decay: 0.0
  max_grad_norm: 1.0
  eval_steps: 10
  save_steps: 10
  logging_steps: 1
  save_total_limit: 1
  best_metric: wer_decoded_norm

eval:
  split: test
  model_dir: null
  out_json: {tmp_path / "eval_outputs" / "metrics.json"}
  out_jsonl: null
  batch_size: 2
  num_workers: 0
  num_samples: 0
  max_sec: 0.0
  discard_number_samples: false
  space_chinese_chars: true
  verbalize_numbers: false
  device: cpu
  columns:
    audio: audio_path
    transcript: text
    id: id
"""
    cfg_path = tmp_path / "test_config.yaml"
    cfg_path.write_text(cfg_text, encoding="utf-8")
    return cfg_path


@pytest.fixture()
def local_cfg(local_config_path):
    """Loaded config dict from the local_config_path fixture."""
    return load_yaml(local_config_path)
