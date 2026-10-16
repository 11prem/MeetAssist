"""Streaming speech segmenter (Silero VAD v6 bundled with faster-whisper).

Feed it 16 kHz mono float32 audio as it arrives; it returns finished utterances cut at
natural pauses. Cut rules: silence >= ``min_silence_s`` after speech; or a force-cut at
``max_s`` placed at the quietest recent point (never mid-word if avoidable).
Timestamps are seconds on the meeting timeline (sample index / 16000).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np

RATE = 16000
FRAME = 512                      # silero v6 frame = 32 ms @ 16 kHz
FRAME_S = FRAME / RATE
CTX = 64                         # context samples silero prepends to every frame


@dataclass
class Utterance:
    track: str
    start_s: float
    end_s: float
    audio: np.ndarray            # float32 mono 16 kHz

    @property
    def dur(self) -> float:
        return self.end_s - self.start_s


_vad = None


def _get_vad():
    global _vad
    if _vad is None:
        from faster_whisper.vad import get_vad_model
        _vad = get_vad_model()
    return _vad


class TrackSegmenter:
    def __init__(self, track: str, *, threshold: float = 0.5, min_silence_s: float = 0.7,
                 min_speech_s: float = 0.30, pad_s: float = 0.20, max_s: float = 28.0) -> None:
        self.track = track
        self.th, self.neg = threshold, max(threshold - 0.15, 0.1)
        self.min_sil = int(min_silence_s / FRAME_S)
        self.min_speech = int(min_speech_s / FRAME_S)
        self.pad = int(pad_s / FRAME_S)
        self.max_frames = int(max_s / FRAME_S)
        self.vad = _get_vad()
        self.buf = np.zeros(0, dtype=np.float32)   # audio starting at absolute frame `fbase`
        self.fbase = 0                             # absolute frame index of buf[0]
        self.probs: List[float] = []               # probs[k] belongs to absolute frame fbase + k
        self.i = 0                                 # next absolute frame to examine
        self.h = np.zeros((1, 1, 128), np.float32)   # silero recurrent state
        self.c = np.zeros((1, 1, 128), np.float32)
        self.ctx = np.zeros(CTX, np.float32)
        self.in_speech = False
        self.sp_start = 0                          # absolute frame where current speech began
        self.last_speech = 0                       # absolute frame of last speech-ish frame

    # ------------------------------------------------------------------ scoring
    def _score_new(self) -> None:
        """Score newly available frames, carrying Silero's recurrent state between calls
        (verified bit-identical to scoring the whole file at once; chunk-local scoring was not)."""
        have = self.fbase + len(self.buf) // FRAME          # frames fully available (absolute count)
        done = self.fbase + len(self.probs)
        if have <= done:
            return
        fr = self.buf[(done - self.fbase) * FRAME:(have - self.fbase) * FRAME].reshape(-1, FRAME)
        ctxs = np.concatenate([self.ctx[None, :], fr[:-1, -CTX:]], 0)
        inp = np.concatenate([ctxs, fr], 1).astype(np.float32)
        out, self.h, self.c = self.vad.session.run(None, {"input": inp, "h": self.h, "c": self.c})
        self.ctx = fr[-1, -CTX:].copy()
        self.probs.extend(np.asarray(out).reshape(-1).tolist())

    # ------------------------------------------------------------------ public
    def feed(self, x: np.ndarray) -> List[Utterance]:
        self.buf = np.concatenate([self.buf, x.astype(np.float32, copy=False)])
        self._score_new()
        return self._scan(final=False)

    def flush(self) -> List[Utterance]:
        """End of stream: pad to a whole frame and emit whatever speech is pending."""
        rem = len(self.buf) % FRAME
        if rem:
            self.buf = np.concatenate([self.buf, np.zeros(FRAME - rem, dtype=np.float32)])
        self._score_new()
        return self._scan(final=True)

    # ------------------------------------------------------------------ state machine
    def _p(self, f: int) -> float:
        return self.probs[f - self.fbase]

    def _scan(self, final: bool) -> List[Utterance]:
        out: List[Utterance] = []
        end = self.fbase + len(self.probs)
        while self.i < end:
            i, p = self.i, self._p(self.i)
            if not self.in_speech:
                if p >= self.th:
                    self.in_speech, self.sp_start, self.last_speech = True, i, i
            else:
                if p >= self.neg:
                    self.last_speech = i
                if i - self.last_speech >= self.min_sil:
                    self._cut(self.sp_start, self.last_speech, out)
                    self.in_speech = False
                elif i - self.sp_start >= self.max_frames:
                    lo = max(self.sp_start + self.min_speech, i - int(8 / FRAME_S))
                    k = lo + int(np.argmin([self._p(f) for f in range(lo, i + 1)]))
                    self._cut(self.sp_start, k, out, forced=True)
                    self._nopad = k + 1                              # next piece starts right after the cut, no lead-in
                    self.sp_start, self.last_speech = k + 1, i      # speech carries on after the cut
            self.i += 1
        if final and self.in_speech:
            self._cut(self.sp_start, self.last_speech, out)
            self.in_speech = False
        self._trim()
        return out

    def _cut(self, s: int, e: int, out: List[Utterance], forced: bool = False) -> None:
        if (e - s + 1) < self.min_speech:
            return
        # a forced cut happens mid-speech: pad neither side of the cut, or the words there would be heard twice
        a = s if s == getattr(self, "_nopad", -1) else max(s - self.pad, self.fbase)
        b = e + 1 if forced else min(e + 1 + self.pad, self.fbase + len(self.probs))
        audio = self.buf[(a - self.fbase) * FRAME:(b - self.fbase) * FRAME].copy()
        out.append(Utterance(self.track, a * FRAME_S, b * FRAME_S, audio))

    def _trim(self) -> None:
        """Drop audio we can never need again (keeps memory bounded on long meetings)."""
        keep = (self.sp_start if self.in_speech else self.i) - self.pad
        drop = keep - self.fbase
        if drop > 50:
            self.buf = self.buf[drop * FRAME:]
            self.probs = self.probs[drop:]
            self.fbase += drop
