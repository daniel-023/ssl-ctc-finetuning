import re
from pathlib import Path
from typing import Optional

TARGET_SR = 16000
TAG_RE = re.compile(r"<[^>]+>")
PUNC_RE = re.compile(r"[\,\?\.\!\-\;\:\“\”\"\%\—\–\…\(\)\[\]\{\}]")


def resolve_path(value, config_path: Path) -> Optional[Path]:
    if value is None:
        return None
    p = Path(str(value))
    if p.is_absolute():
        return p
    return (config_path.parent / p).resolve()


def normalize_text_basic(t: str) -> str:
    t = str(t or "").lower()
    t = TAG_RE.sub(" ", t)
    t = PUNC_RE.sub("", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


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
        Normalizer(
            cfg,
            enable_chinese_spacing=enable_chinese_spacing,
            verbalize_numbers=verbalize_numbers,
        ),
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
