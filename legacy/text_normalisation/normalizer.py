#!/usr/bin/env python3
"""
Usage:
  # Default: split Chinese text into single-character tokens, keep rows with numbers
  python normalizer.py --yaml fillers.yaml --jsonl in.jsonl --out_jsonl out.jsonl

  # Disable Chinese character spacing (keep contiguous Chinese runs)
  python normalizer.py --yaml fillers.yaml --jsonl in.jsonl --out_jsonl out.jsonl --no-space-chinese-chars

  # Discard samples containing any digits
  python normalizer.py --yaml fillers.yaml --jsonl in.jsonl --out_jsonl out.jsonl --discard-number-samples

  # Verbalize number tokens using utterance dominant language (en/zh)
  python normalizer.py --yaml fillers.yaml --jsonl in.jsonl --out_jsonl out.jsonl --verbalize-numbers
"""

import argparse
import json
import re
import unicodedata
from pathlib import Path
from typing import Dict, List, Set, Tuple
import yaml
import cn2an
from num2words import num2words


# Config
class NormalizerConfig:
    def __init__(self, yaml_path: str):
        data = yaml.safe_load(Path(yaml_path).read_text(encoding="utf-8"))
        self.remove_tags = set(data.get("remove_tags", [])) or set()
        options = data.get("options", {}) or {}
        self.lowercase = bool(options.get("lowercase", True))
        self.strip_punctuation = bool(options.get("strip_punctuation", True))
        self.remove_angle_tags = bool(options.get("remove_angle_tags", True))
        self.remove_bracket_tags = bool(options.get("remove_bracket_tags", True))
        self.collapse_whitespace = bool(options.get("collapse_whitespace", True))

        self.filler_map, self.canonical_fillers = self._parse_fillers(data.get("fillers", {}) or {})
        self.language_aware_fillers = self._parse_language_aware_fillers(
            data.get("language_aware_fillers", {}) or {}
        )

    def _parse_fillers(self, fillers_block: Dict) -> Tuple[Dict[str, str], Set[str]]:
        filler_map: Dict[str, str] = {}
        canonical_fillers: Set[str] = set()
        self.non_removable_fillers: Set[str] = set()

        for canonical, data in fillers_block.items():
            if canonical == "non_removable":
                continue

            canonical = str(canonical)
            canonical_fillers.add(canonical)

            # Handle variants and non_removable flag
            if isinstance(data, dict):
                variants = data.get("variants", [])
                if data.get("non_removable", False):
                    self.non_removable_fillers.add(canonical)
            else:
                variants = data

            filler_map[canonical] = canonical
            for variant in variants or []:
                filler_map[str(variant)] = canonical

        return filler_map, canonical_fillers

    def _parse_language_aware_fillers(self, block: Dict) -> Dict[str, Dict[str, str]]:
        out: Dict[str, Dict[str, str]] = {}
        for token, mapping in block.items():
            if not isinstance(mapping, dict):
                continue
            key = str(token)
            out[key] = {}
            for lang in ("en", "zh"):
                value = mapping.get(lang)
                if value is not None:
                    out[key][lang] = str(value)
        return out

# Normalisation
CJK_RANGE = r"\u4E00-\u9FFF\u3400-\u4DBF\uF900-\uFAFF"
TOKEN_RE_CHAR_SPACING = re.compile(
    rf"([{CJK_RANGE}])|([a-zA-Z]+(?:'[a-zA-Z]+)?)|(\d+(?:\.\d+)?)"
)
TOKEN_RE_PRESERVE_CJK_RUNS = re.compile(
    rf"([{CJK_RANGE}]+)|([a-zA-Z]+(?:'[a-zA-Z]+)?)|(\d+(?:\.\d+)?)"
)

ANGLE_TAG_RE = re.compile(r"<[^>]*>")
BRACKET_TAG_RE = re.compile(r"\[[^\]]*\]")

# Remove punctuation but keep letters/digits/CJK/space/apostrophe
PUNCT_RE = re.compile(rf"[^0-9a-zA-Z{CJK_RANGE}\s']+")
LATIN_WORD_RE = re.compile(r"[a-zA-Z]+(?:'[a-zA-Z]+)?")
DIGIT_RE = re.compile(r"\d")
NUMERIC_TOKEN_RE = re.compile(r"\d+(?:\.\d+)?")


class Normalizer:
    def __init__(
        self,
        config: NormalizerConfig,
        enable_chinese_spacing: bool = True,
        verbalize_numbers: bool = False,
    ):
        self.cfg = config
        self.enable_chinese_spacing = enable_chinese_spacing
        self.verbalize_numbers = verbalize_numbers

    def normalize(self, text: str) -> Dict[str, object]:
        s = self._clean(text)
        toks = self._tokenize(s)
        dominant_language = self._detect_dominant_language(toks)

        toks_canon, fillers_canon, fillers_raw, removable_mask = self._canonicalize_fillers(
            toks, dominant_language
        )

        toks_norm, toks_no_fill = self._render_tokens(toks_canon, removable_mask, dominant_language)

        return {
            "text_norm": " ".join(toks_norm),
            "text_no_fill": " ".join(toks_no_fill),
            "fillers": fillers_raw,       
            "fillers_canon": fillers_canon,  
        }

    def _clean(self, text: str) -> str:
        s = unicodedata.normalize("NFKC", str(text or ""))
        s = s.replace("’", "'").replace("‘", "'")

        # exact tag removal
        for tag in self.cfg.remove_tags:
            s = s.replace(tag, " ")

        # generic tag removal
        if self.cfg.remove_angle_tags:
            s = ANGLE_TAG_RE.sub(" ", s)
        if self.cfg.remove_bracket_tags:
            s = BRACKET_TAG_RE.sub(lambda m: m.group(0)[1:-1], s)  # Remove brackets but keep content

        if self.cfg.lowercase:
            s = s.lower()

        if self.cfg.strip_punctuation:
            s = PUNCT_RE.sub(" ", s)

        if self.cfg.collapse_whitespace:
            s = re.sub(r"\s+", " ", s).strip()

        return s

    def _detect_dominant_language(self, tokens: List[str]) -> str:
        # Compare Chinese script volume (characters) vs English lexical volume (words).
        cjk_count = 0
        latin_count = 0
        for token in tokens:
            cjk_count += len(re.findall(rf"[{CJK_RANGE}]", token))
            if LATIN_WORD_RE.fullmatch(token):
                latin_count += 1

        if cjk_count == 0 and latin_count == 0:
            return "neutral"

        # Require a small margin so mixed code-switching stays neutral.
        if cjk_count >= latin_count * 1.2:
            return "zh"
        if latin_count >= cjk_count * 1.2:
            return "en"
        return "neutral"

    def _tokenize(self, s: str) -> List[str]:
        toks: List[str] = []
        token_re = TOKEN_RE_CHAR_SPACING if self.enable_chinese_spacing else TOKEN_RE_PRESERVE_CJK_RUNS
        for m in token_re.finditer(s):
            cjk, lat, num = m.groups()
            if cjk:
                toks.append(cjk)
            elif lat:
                toks.append(lat)
            elif num:
                toks.append(num)
        return toks

    def _canonicalize_fillers(self, tokens, dominant_language: str):
        fillers_canon = []
        fillers_raw = []
        out = []
        removable_mask = []

        removable = self.cfg.canonical_fillers - self.cfg.non_removable_fillers

        for token in tokens:
            base_canonical = self.cfg.filler_map.get(token, token)
            canonical = base_canonical
            lang_map = self.cfg.language_aware_fillers.get(token)
            if not lang_map:
                lang_map = self.cfg.language_aware_fillers.get(canonical)
            if lang_map:
                canonical = lang_map.get(dominant_language, canonical)
            out.append(canonical)

            # Keep filler identity tied to the base canonical class so language-aware
            # remapping (e.g., lah -> 啦) does not break filler removal/tracking.
            is_removable = base_canonical in removable or canonical in removable
            removable_mask.append(is_removable)
            if is_removable:
                fillers_canon.append(canonical)
                fillers_raw.append(token)

        return out, fillers_canon, fillers_raw, removable_mask

    def _render_tokens(self, tokens: List[str], removable_mask: List[bool], dominant_language: str):
        toks_norm: List[str] = []
        toks_no_fill: List[str] = []
        for token, is_removable in zip(tokens, removable_mask):
            expanded = self._expand_token(token, dominant_language)
            toks_norm.extend(expanded)
            if not is_removable:
                toks_no_fill.extend(expanded)
        return toks_norm, toks_no_fill

    def _expand_token(self, token: str, dominant_language: str) -> List[str]:
        if not self.verbalize_numbers:
            return [token]
        if dominant_language not in ("en", "zh"):
            return [token]
        if not NUMERIC_TOKEN_RE.fullmatch(token):
            return [token]
        if len(token) > 1 and token.startswith("0") and not token.startswith("0."):
            return [token]

        if dominant_language == "en":
            spoken = self._number_to_english(token)
            return spoken if spoken else [token]
        spoken = self._number_to_chinese(token)
        return spoken if spoken else [token]

    def _number_to_english(self, token: str) -> List[str]:
        if num2words is None:
            return []
        try:
            if "." in token:
                int_part, frac_part = token.split(".", 1)
                int_words = num2words(int(int_part))
                frac_words = " ".join(num2words(int(ch)) for ch in frac_part)
                spoken = f"{int_words} point {frac_words}"
            else:
                spoken = num2words(int(token))
        except Exception:
            return []
        return LATIN_WORD_RE.findall(spoken.lower())

    def _number_to_chinese(self, token: str) -> List[str]:
        if cn2an is None:
            return []
        try:
            spoken = str(cn2an.an2cn(token)).replace(" ", "")
        except Exception:
            return []
        if self.enable_chinese_spacing:
            return [ch for ch in spoken if not ch.isspace()]
        return [spoken] if spoken else []

# JSONL Processing
def process_jsonl(
    in_path: str,
    out_path: str,
    normalizer: Normalizer,
    discard_number_samples: bool = False,
):
    in_p = Path(in_path)
    out_p = Path(out_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)

    with in_p.open("r", encoding="utf-8") as fin, out_p.open("w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)

            ref = obj.get("ref", "")
            hyp = obj.get("hyp")  # Keep hyp as None if not present

            if discard_number_samples:
                if DIGIT_RE.search(str(ref or "")) or DIGIT_RE.search(str(hyp or "")):
                    continue

            ref_n = normalizer.normalize(ref)
            obj["ref_norm"] = ref_n["text_norm"]
            obj["ref_no_fill"] = ref_n["text_no_fill"]
            obj["ref_fillers"] = list(set(ref_n["fillers"]))

            if hyp is not None:
                hyp_n = normalizer.normalize(hyp)
                obj["hyp_norm"] = hyp_n["text_norm"]
                obj["hyp_no_fill"] = hyp_n["text_no_fill"]
                if hyp_n["fillers"]:
                    obj["hyp_fillers"] = hyp_n["fillers"]


            fout.write(json.dumps(obj, ensure_ascii=False) + "\n")


# CLI
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--yaml", default="fillers.yaml", help="Path to fillers.yaml")
    parser.add_argument("--jsonl", default="sample.jsonl", help="Input JSONL file")
    parser.add_argument("--out_jsonl", default="normalized_sample.jsonl", help="Output JSONL file")
    parser.add_argument(
        "--space-chinese-chars",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="If true, split Chinese text into single-character tokens; if false, keep contiguous Chinese runs.",
    )
    parser.add_argument(
        "--discard-number-samples",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="If true, skip JSONL rows where ref or hyp contains any digits.",
    )
    parser.add_argument(
        "--verbalize-numbers",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="If true, verbalize numeric tokens using utterance dominant language (en/zh).",
    )
    args = parser.parse_args()

    config = NormalizerConfig(args.yaml)
    normalizer = Normalizer(
        config,
        enable_chinese_spacing=args.space_chinese_chars,
        verbalize_numbers=args.verbalize_numbers,
    )
    process_jsonl(
        args.jsonl,
        args.out_jsonl,
        normalizer,
        discard_number_samples=args.discard_number_samples,
    )


if __name__ == "__main__":
    main()
