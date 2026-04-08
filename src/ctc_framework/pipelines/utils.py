from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

TARGET_SR = 16000


def resolve_path(value, config_path: Path) -> Optional[Path]:
    if value is None:
        return None
    p = Path(str(value))
    if p.is_absolute():
        return p
    return (config_path.parent / p).resolve()


def build_text_normalizer(
    normalizer_yaml: Optional[str],
    config_path: Path,
    *,
    enable_chinese_spacing: bool = True,
    verbalize_numbers: bool = False,
):
    from ctc_framework.text.normalizer import Normalizer, NormalizerConfig

    if normalizer_yaml is None:
        yaml_path = Path(__file__).resolve().parents[1] / "text" / "fillers.yaml"
    else:
        yaml_path = resolve_path(normalizer_yaml, config_path)
        if yaml_path is None:
            raise ValueError("normalizer_yaml resolution failed")

    if not yaml_path.exists():
        raise FileNotFoundError(f"Normalizer YAML not found: {yaml_path}")

    cfg = NormalizerConfig(str(yaml_path))
    return (
        Normalizer(cfg, enable_chinese_spacing=enable_chinese_spacing, verbalize_numbers=verbalize_numbers),
        str(yaml_path),
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


def build_normalizer_fn(
    normalizer_yaml,
    config_path: Path,
    *,
    enable_chinese_spacing: bool = True,
    verbalize_numbers: bool = False,
) -> Tuple[Callable[[str], str], Callable[[str], Tuple[str, str]], str]:
    """Build text normalization callables using the full normalizer.

    Returns:
        text_norm_fn: normalizes text to a single string (text_norm).
        normalize_both_fn: returns (text_norm, text_no_fill) — use in eval for filler-free WER.
        used_yaml: path to the normalizer YAML that was loaded.
    """
    normalizer, used_yaml = build_text_normalizer(
        normalizer_yaml,
        config_path,
        enable_chinese_spacing=enable_chinese_spacing,
        verbalize_numbers=verbalize_numbers,
    )
    print(f"Using normalizer: {used_yaml}")

    def text_norm_fn(s: str) -> str:
        return normalizer.normalize(s)["text_norm"]  # type: ignore[index]

    def normalize_both_fn(s: str) -> Tuple[str, str]:
        out = normalizer.normalize(s)
        return str(out["text_norm"]), str(out["text_no_fill"])

    return text_norm_fn, normalize_both_fn, used_yaml


def load_jsonl_transcript_map(cfg: dict, config_path: Path) -> Dict[str, str]:
    """Load a JSONL transcript file into an id→text mapping.

    Reads transcript.join.json_key and transcript.jsonl.text_col from config.
    Raises on missing file, missing columns, or duplicate keys.
    """
    from datasets import load_dataset as _load_dataset
    from ctc_framework.config.loader import get_in

    json_path = resolve_path(get_in(cfg, "transcript.jsonl.json_path"), config_path)
    if json_path is None or not json_path.exists():
        raise FileNotFoundError(f"Transcript JSON not found: {json_path}")

    id_col = str(get_in(cfg, "transcript.join.json_key", "id"))
    text_col = str(get_in(cfg, "transcript.jsonl.text_col", "text"))

    ds = _load_dataset("json", data_files=str(json_path), split="train")
    if id_col not in ds.column_names:
        raise ValueError(
            f"Transcript id column '{id_col}' not found in {json_path}. "
            f"Available columns: {ds.column_names}"
        )
    if text_col not in ds.column_names:
        raise ValueError(
            f"Transcript text column '{text_col}' not found in {json_path}. "
            f"Available columns: {ds.column_names}"
        )

    out: Dict[str, str] = {}
    for row in ds:
        key = str(row[id_col])
        val = str(row[text_col] or "")
        if key in out:
            raise ValueError(f"Duplicate transcript key '{key}' found in {json_path}.")
        out[key] = val

    if not out:
        raise ValueError(f"Transcript JSON is empty: {json_path}")
    return out


def apply_jsonl_transcripts(
    ds,
    split_name: str,
    cfg: dict,
    transcript_map: Dict[str, str],
    *,
    text_col: Optional[str] = None,
):
    """Join a transcript map onto a dataset by utterance ID.

    Reads transcript.join.dataset_key and transcript.join.strict from config.
    If text_col is None, reads dataset.columns.transcript from config.
    Unmatched rows are dropped (or raise if strict=True).
    Always adds/overwrites a __utt_id column with the matched IDs.
    """
    from ctc_framework.config.loader import get_in

    if text_col is None:
        text_col = str(get_in(cfg, "dataset.columns.transcript"))
    dataset_id_col = str(get_in(cfg, "transcript.join.dataset_key", "id"))
    strict = bool(get_in(cfg, "transcript.join.strict", True))

    if dataset_id_col not in ds.column_names:
        raise ValueError(
            f"Dataset id column '{dataset_id_col}' not found for split '{split_name}'. "
            f"Available columns: {ds.column_names}"
        )

    ids = [str(x) for x in ds[dataset_id_col]]
    keep_indices: List[int] = []
    keep_texts: List[str] = []
    keep_ids: List[str] = []
    missing: List[str] = []

    for i, key in enumerate(ids):
        text = transcript_map.get(key)
        if text is None:
            missing.append(key)
            continue
        keep_indices.append(i)
        keep_texts.append(text)
        keep_ids.append(key)

    if not keep_indices:
        raise ValueError(
            f"Transcript JSON join produced 0 matches for split '{split_name}'. "
            "Check dataset/transcript join keys and id formats."
        )

    if missing and strict:
        preview = ", ".join(missing[:5])
        raise ValueError(
            f"Transcript JSON join missing {len(missing)} ids for split '{split_name}'. "
            f"Examples: {preview}. Set transcript.join.strict=false to drop unmatched rows."
        )

    out_ds = ds.select(keep_indices)
    if text_col in out_ds.column_names:
        out_ds = out_ds.remove_columns(text_col)
    out_ds = out_ds.add_column(text_col, keep_texts)

    if "__utt_id" in out_ds.column_names:
        out_ds = out_ds.remove_columns("__utt_id")
    out_ds = out_ds.add_column("__utt_id", keep_ids)

    print(
        f"Transcript JSON join split={split_name}: matched={len(keep_indices)} "
        f"dropped_unmatched={len(missing)} source_rows={len(ds)}"
    )
    return out_ds


def maybe_prefix_local_audio_paths(ds, audio_col: str, audio_root: Optional[Path], num_proc: int = 1):
    if audio_root is None or audio_col not in ds.column_names or len(ds) == 0:
        return ds

    sample = ds[0][audio_col]
    if not isinstance(sample, str):
        return ds

    if not audio_root.exists():
        raise FileNotFoundError(f"Local audio root not found: {audio_root}")

    def _prefix(ex):
        value = ex[audio_col]
        if value is None:
            return {audio_col: value}
        p = Path(str(value))
        if p.is_absolute():
            return {audio_col: str(p)}
        return {audio_col: str((audio_root / p).resolve())}

    return ds.map(_prefix, num_proc=num_proc)
