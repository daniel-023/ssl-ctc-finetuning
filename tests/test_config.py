"""Tests for ctc_framework.config.loader and schema defaults."""

import pytest

from ctc_framework.config.loader import apply_overrides, get_in, load_yaml
from ctc_framework.config.schema import DEFAULTS


# ---------------------------------------------------------------------------
# Schema defaults
# ---------------------------------------------------------------------------

def test_schema_has_required_training_keys():
    training = DEFAULTS["training"]
    for key in ("bs", "grad_accum", "lr", "epochs", "eval_steps", "save_steps", "num_proc", "seed"):
        assert key in training, f"Missing training key: {key}"


def test_schema_has_required_eval_keys():
    eval_ = DEFAULTS["eval"]
    for key in ("split", "batch_size", "num_workers", "device", "out_json"):
        assert key in eval_, f"Missing eval key: {key}"


def test_schema_no_use_text_normalizer():
    """use_text_normalizer was removed — ensure it is not in the schema."""
    assert "use_text_normalizer" not in DEFAULTS.get("normalization", {})
    assert "use_text_normalizer" not in DEFAULTS.get("eval", {})


# ---------------------------------------------------------------------------
# load_yaml — defaults merge
# ---------------------------------------------------------------------------

def test_load_yaml_merges_defaults(local_config_path):
    cfg = load_yaml(local_config_path)
    # Schema defaults should be present even if not in the YAML
    assert "model" in cfg
    assert cfg["model"]["name_or_path"] == "facebook/wav2vec2-xls-r-300m"


def test_load_yaml_user_value_overrides_default(local_config_path):
    cfg = load_yaml(local_config_path)
    # Config fixture sets backend=local, which overrides default "hf"
    assert cfg["dataset"]["backend"] == "local"


def test_load_yaml_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        load_yaml("/nonexistent/path/config.yaml")


# ---------------------------------------------------------------------------
# get_in
# ---------------------------------------------------------------------------

def test_get_in_nested():
    cfg = {"a": {"b": {"c": 42}}}
    assert get_in(cfg, "a.b.c") == 42


def test_get_in_missing_returns_default():
    cfg = {"a": {}}
    assert get_in(cfg, "a.b.c", "fallback") == "fallback"


def test_get_in_none_default():
    cfg = {}
    assert get_in(cfg, "nonexistent") is None


# ---------------------------------------------------------------------------
# apply_overrides
# ---------------------------------------------------------------------------

def test_apply_overrides_int(local_config_path):
    cfg = load_yaml(local_config_path)
    out = apply_overrides(cfg, ["training.bs=4"])
    assert get_in(out, "training.bs") == 4


def test_apply_overrides_float(local_config_path):
    cfg = load_yaml(local_config_path)
    out = apply_overrides(cfg, ["training.lr=0.001"])
    assert abs(get_in(out, "training.lr") - 0.001) < 1e-10


def test_apply_overrides_string(local_config_path):
    cfg = load_yaml(local_config_path)
    out = apply_overrides(cfg, ["training.out_dir=/tmp/myrun"])
    assert get_in(out, "training.out_dir") == "/tmp/myrun"


def test_apply_overrides_bool_true(local_config_path):
    cfg = load_yaml(local_config_path)
    out = apply_overrides(cfg, ["eval.discard_number_samples=true"])
    assert get_in(out, "eval.discard_number_samples") is True


def test_apply_overrides_bool_false(local_config_path):
    cfg = load_yaml(local_config_path)
    out = apply_overrides(cfg, ["eval.discard_number_samples=false"])
    assert get_in(out, "eval.discard_number_samples") is False


def test_apply_overrides_null(local_config_path):
    cfg = load_yaml(local_config_path)
    out = apply_overrides(cfg, ["eval.model_dir=null"])
    assert get_in(out, "eval.model_dir") is None


def test_apply_overrides_does_not_mutate_original(local_config_path):
    cfg = load_yaml(local_config_path)
    original_bs = get_in(cfg, "training.bs")
    apply_overrides(cfg, ["training.bs=999"])
    assert get_in(cfg, "training.bs") == original_bs


def test_apply_overrides_invalid_format(local_config_path):
    cfg = load_yaml(local_config_path)
    with pytest.raises(ValueError):
        apply_overrides(cfg, ["no_equals_sign"])


def test_apply_overrides_multiple(local_config_path):
    cfg = load_yaml(local_config_path)
    out = apply_overrides(cfg, ["training.bs=8", "training.epochs=5"])
    assert get_in(out, "training.bs") == 8
    assert get_in(out, "training.epochs") == 5.0
