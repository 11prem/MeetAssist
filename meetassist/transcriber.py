"""Local ASR: faster-whisper (CTranslate2, int8) with utterance *packing*.

Why packing: the Whisper encoder always processes a full 30 s window (~9.5 s of CPU on the
dev laptop for large-v3-turbo) no matter how little speech is inside. Meetings consist of many
short utterances, so we concatenate consecutive utterances of one track (<= PACK_S seconds of
audio, small silent gaps between) into ONE window, decode once, then map segment timestamps
back onto the real meeting timeline. Measured: removes most of the fixed encoder cost.

Language handling ("mixed" default): Whisper forced to English + a Tanglish primer prompt keeps
Tamil speech as *romanised Tanglish* and English as English (auto-detect translated Tanglish into
English and Tamil-script mode was much worse — see todo.md V2 log).
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

import numpy as np

from .segmenter import RATE, Utterance

PACK_S = 27.0            # max audio per decoder window (Whisper window is 30 s)
GAP_S = 0.4              # silence inserted between packed utterances

TANGLISH_PRIMER = (
    "Vanakkam everyone. Inniku namma meeting la project status pathi pesalam. "
    "Machine learning model eppadi work pannum nu paapom, deadline next week, action items note pannunga."
)

MODEL_DIRS = {
    "large-v3-turbo": "ct2-large-v3-turbo",
    "medium": "ct2-medium",
    "small": "ct2-small",
}

_HALLUCINATIONS = re.compile(
    r"^(thanks? for watching[.!]*|thank you[.!]*|you[.!]*|bye[.!]*|\.+|subtitles? by .*|please subscribe.*)$", re.I)


@dataclass
class Piece:
    track: str
    t_start: float
    t_end: float
    text: str
    lang: str
    alt: str = ""          # the other engine's reading of the same audio (hybrid mode)
    en: str = ""           # English translation of a Tamil piece (hybrid mode)
    echo: bool = False     # microphone picked up the speakers (not the user) - hidden by the reader


def available_models(models_dir: Path) -> List[str]:
    return [k for k, d in MODEL_DIRS.items() if (models_dir / d / "model.bin").exists()]


def _collapse_repeats(text: str) -> str:
    """Whisper sometimes loops ("foo foo foo foo ..."). Collapse runs of the same word/phrase."""
    words = text.split()
    out: List[str] = []
    for w in words:
        if len(out) >= 3 and out[-1] == w and out[-2] == w and out[-3] == w:
            continue
        out.append(w)
    t = " ".join(out)
    t = re.sub(r"(\b[\w']+)(?:[-, ]+\1\b){3,}", r"\1", t)                 # "one-one-one-one", "no, no, no, no"
    return re.sub(r"((?:\b[\w']+\b[ ,]+){2,6}?)\1{3,}", r"\1", t)         # repeated short phrases


class Transcriber:
    def __init__(self, model_dir: Path, threads: int = 4, compute_type: str = "int8", beam: int = 1) -> None:
        from faster_whisper import WhisperModel
        self.model_dir = Path(model_dir)
        self.name = self.model_dir.name
        self.beam = beam
        self.model = WhisperModel(str(model_dir), device="cpu", compute_type=compute_type, cpu_threads=threads,
                                  local_files_only=True)

    # ---------------------------------------------------------------- packing
    @staticmethod
    def pack(utts: Sequence[Utterance], limit_s: float = PACK_S) -> List[List[Utterance]]:
        """Greedy: consecutive utterances (already in time order, same track) into <= limit_s windows."""
        packs: List[List[Utterance]] = []
        cur: List[Utterance] = []
        total = 0.0
        for u in utts:
            need = u.dur + (GAP_S if cur else 0)
            if cur and total + need > limit_s:
                packs.append(cur); cur, total = [], 0.0; need = u.dur
            cur.append(u); total += need
        if cur:
            packs.append(cur)
        return packs

    def _token_cap(self, dur_s: float, prompt: Optional[str]) -> int:
        """Bound the decode length (stops runaway loops) while respecting Whisper's 448-token limit."""
        try:
            n_prompt = len(self.model.hf_tokenizer.encode(prompt, add_special_tokens=False).ids) + 4 if prompt else 0
        except Exception:
            n_prompt = len(prompt or "") // 2 + 4
        return max(48, min(int(dur_s * 14) + 40, 440 - min(n_prompt, 223)))

    # ---------------------------------------------------------------- decode
    def transcribe_pack(self, utts: Sequence[Utterance], language: str = "mixed", vocabulary: str = "",
                        context: str = "") -> List[Piece]:
        """Decode one packed window; returns pieces mapped to the real timeline."""
        gap = np.zeros(int(GAP_S * RATE), dtype=np.float32)
        parts, spans, t = [], [], 0.0                       # spans: (pack_start, pack_end, utterance)
        for k, u in enumerate(utts):
            if k:
                parts.append(gap); t += GAP_S
            parts.append(u.audio); spans.append((t, t + u.dur, u)); t += u.dur
        audio = np.concatenate(parts)

        lang_arg: Optional[str]
        if language in ("mixed", "en"):
            lang_arg = "en"
        elif language == "ta":
            lang_arg = "ta"
        else:
            lang_arg = None                                  # true auto-detect
        prompt = " ".join(x for x in ((vocabulary or "").strip(),
                                      TANGLISH_PRIMER if language == "mixed" else "",
                                      context.strip()[-220:]) if x) or None
        segs, info = self.model.transcribe(
            audio, language=lang_arg, beam_size=self.beam, temperature=[0.0, 0.3],
            condition_on_previous_text=False, initial_prompt=prompt, vad_filter=False,
            no_speech_threshold=0.6, compression_ratio_threshold=2.2, log_prob_threshold=-1.2,
            word_timestamps=False, suppress_blank=True,
            max_new_tokens=self._token_cap(audio.shape[0] / RATE, prompt), repetition_penalty=1.05)

        def to_real(pt: float, is_end: bool) -> float:
            for a, b, u in spans:
                if pt < b + GAP_S / 2 or (a, b, u) is spans[-1]:
                    return min(max(u.start_s + (pt - a), u.start_s), u.end_s)
            return spans[-1][2].end_s

        out: List[Piece] = []
        for s in segs:
            text = _collapse_repeats(s.text.strip())
            if not text or _HALLUCINATIONS.match(text):
                continue
            first = next((u for a, b, u in spans if s.start < b + GAP_S / 2), spans[-1][2])
            out.append(Piece(first.track, to_real(s.start, False), to_real(s.end, True), text, info.language))
        return out

    def transcribe_utterances(self, utts: Sequence[Utterance], vocabulary: str = "", context: str = "") -> List[str]:
        """Decode one packed window with word timestamps and hand the words back to the utterance they
        were spoken in. Returns one string per utterance (possibly empty). Used by the hybrid engine."""
        gap = np.zeros(int(GAP_S * RATE), dtype=np.float32)
        parts, spans, t = [], [], 0.0
        for k, u in enumerate(utts):
            if k:
                parts.append(gap); t += GAP_S
            parts.append(u.audio); spans.append((t, t + u.dur)); t += u.dur
        audio = np.concatenate(parts)
        prompt = " ".join(x for x in ((vocabulary or "").strip(), TANGLISH_PRIMER, context.strip()[-220:]) if x) or None
        segs, _info = self.model.transcribe(
            audio, language="en", beam_size=self.beam, temperature=[0.0, 0.3], condition_on_previous_text=False,
            initial_prompt=prompt, vad_filter=False, no_speech_threshold=0.6, compression_ratio_threshold=2.2,
            log_prob_threshold=-1.2, word_timestamps=True, suppress_blank=True,
            max_new_tokens=self._token_cap(audio.shape[0] / RATE, prompt), repetition_penalty=1.05)
        words: List[List[str]] = [[] for _ in utts]
        for s in segs:
            for w in (s.words or []):
                mid = (w.start + w.end) / 2
                k = next((i for i, (a, b) in enumerate(spans) if mid < b + GAP_S / 2), len(spans) - 1)
                words[k].append(w.word)
        out = []
        for ws in words:
            txt = _collapse_repeats("".join(ws).strip())
            out.append("" if _HALLUCINATIONS.match(txt) else txt)
        return out

    def translate_utterances(self, utts: Sequence[Utterance], vocabulary: str = "", context: str = "") -> List[str]:
        """Whisper 'translate' task (Tamil/Tanglish -> English) over one packed window; words are handed back to the
        utterance they were spoken in. One string per utterance (possibly empty)."""
        gap = np.zeros(int(GAP_S * RATE), dtype=np.float32)
        parts, spans, t = [], [], 0.0
        for k, u in enumerate(utts):
            if k:
                parts.append(gap); t += GAP_S
            parts.append(u.audio); spans.append((t, t + u.dur)); t += u.dur
        audio = np.concatenate(parts)
        prompt = " ".join(x for x in ((vocabulary or "").strip(), context.strip()[-200:]) if x) or None
        segs, _info = self.model.transcribe(
            audio, task="translate", language="ta", beam_size=self.beam, temperature=[0.0, 0.3],
            condition_on_previous_text=False, initial_prompt=prompt, vad_filter=False, no_speech_threshold=0.6,
            compression_ratio_threshold=2.2, log_prob_threshold=-1.2, word_timestamps=True, suppress_blank=True,
            max_new_tokens=self._token_cap(audio.shape[0] / RATE, prompt), repetition_penalty=1.05)
        words: List[List[str]] = [[] for _ in utts]
        for s in segs:
            for w in (s.words or []):
                mid = (w.start + w.end) / 2
                k = next((i for i, (a, b) in enumerate(spans) if mid < b + GAP_S / 2), len(spans) - 1)
                words[k].append(w.word)
        out = []
        for ws in words:
            txt = _collapse_repeats("".join(ws).strip())
            out.append("" if _HALLUCINATIONS.match(txt) else txt)
        return out
