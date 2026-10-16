"""Per-utterance English-vs-Tamil routing for the hybrid engine (no model needed)."""
from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path

from . import tamil


@lru_cache(maxsize=1)
def _words() -> frozenset:
    return frozenset(Path(__file__).with_name("data").joinpath("english_words.txt").read_text(encoding="utf-8").split())


def english_ratio(text: str) -> float | None:
    """Share of Latin words (>=3 letters) that are real English words; None if there are none."""
    ws = [w.lower() for w in re.findall(r"[A-Za-z']+", text) if len(w.strip("'")) >= 3]
    if not ws:
        return None
    W = _words()
    return sum(w.strip("'") in W for w in ws) / len(ws)


def is_english(whisper_text: str, indic_text: str) -> bool:
    """True when the utterance is English speech (use Whisper's text), False when it is Tamil/Tanglish
    (use IndicConformer's Tamil text).  Whisper forced to English reproduces English cleanly (>=0.9 of
    words are dictionary words) but writes Tamil as phonetic gibberish (<=0.56); Tamil particles in the
    Indic output are a second, independent veto."""
    if tamil.tamil_marker_count(indic_text) >= 1:
        return False
    r = english_ratio(whisper_text)
    if r is None:
        return False
    n_words = len(re.findall(r"[A-Za-z']+", whisper_text))
    return r >= 0.8 if n_words >= 3 else r >= 0.99
