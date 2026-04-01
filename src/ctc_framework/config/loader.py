import copy
from pathlib import Path
from typing import Any

import yaml

from .schema import DEFAULTS


def load_yaml(path: str | Path) -> dict:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config file not found: {p}")
    data = yaml.safe_load(p.read_text(encoding="utf-8"))
    if data is None:
        data = {}
    if not isinstance(data, dict):
        raise ValueError("Top-level config must be a mapping/object.")
    merged = copy.deepcopy(DEFAULTS)
    _deep_update(merged, data)
    return merged


def save_yaml(path: str | Path, payload: dict) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(yaml.safe_dump(payload, sort_keys=False, allow_unicode=True), encoding="utf-8")


def apply_overrides(cfg: dict, overrides: list[str]) -> dict:
    out = copy.deepcopy(cfg)
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"Override must be key=value, got: {item}")
        key, raw_value = item.split("=", 1)
        _set_in(out, key.strip().split("."), _parse_scalar(raw_value.strip()))
    return out


def get_in(cfg: dict, dotted: str, default: Any = None) -> Any:
    cur: Any = cfg
    for part in dotted.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur


def _deep_update(base: dict, patch: dict) -> None:
    for k, v in patch.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _deep_update(base[k], v)
        else:
            base[k] = v


def _set_in(cfg: dict, keys: list[str], value: Any) -> None:
    cur = cfg
    for k in keys[:-1]:
        if k not in cur or not isinstance(cur[k], dict):
            cur[k] = {}
        cur = cur[k]
    cur[keys[-1]] = value


def _parse_scalar(raw: str) -> Any:
    low = raw.lower()
    if low in {"true", "false"}:
        return low == "true"
    if low in {"none", "null"}:
        return None
    try:
        if "." in raw:
            return float(raw)
        return int(raw)
    except ValueError:
        return raw
