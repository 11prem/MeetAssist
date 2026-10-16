"""ASR worker process: tails <meeting>/raw/*.pcm -> VAD segmenter -> engine -> journal.jsonl

Runs detached from the GUI at BELOW_NORMAL priority (``python -m meetassist.asr_worker``), so capture and the
user's Teams/Webex call are never starved, and a GUI crash loses nothing. Also used for file import,
re-transcription and background refinement (same raw/ layout, END marker already present).

Engines (``--language``):
  hybrid   (default) every utterance -> IndicConformer (Tamil script, fast, very good Tamil) AND Whisper (packed,
           very good English). Per utterance, lid.is_english() picks which text is kept; the other is stored as `alt`.
  ta       IndicConformer only (Tamil script; English words are transliterated).   fastest, lowest latency
  en       Whisper only, English.
  whisper  Whisper only, forced-English decoding with a Tanglish primer -> romanised Tanglish.

Files (all inside the meeting dir):
  raw/mic.pcm raw/loop.pcm   input (written by capture_proc, or by import)
  raw/END                    "no more audio will arrive" - written by the GUI after capture has exited
  journal.jsonl              output, one fsynced line per committed piece (torn last line is tolerated)
  worker.json                progress/lag for the GUI; state = loading|listening|draining|done|error
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Dict, List

import numpy as np

from . import echo, lid, paths, tamil
from .segmenter import RATE, TrackSegmenter, Utterance
from .transcriber import MODEL_DIRS, Piece, Transcriber

TRACKS = ("mic", "loop")
MODES = ("hybrid", "ta", "en", "whisper")


class PcmTail:
    """Reads newly appended int16 samples from a growing file."""

    def __init__(self, path: Path) -> None:
        self.path, self.pos = path, 0

    def read(self) -> np.ndarray:
        try:
            size = self.path.stat().st_size
        except FileNotFoundError:
            return np.zeros(0, np.float32)
        n = (size - self.pos) // 2
        if n <= 0:
            return np.zeros(0, np.float32)
        with open(self.path, "rb") as f:
            f.seek(self.pos); data = f.read(n * 2)
        data = data[: len(data) - (len(data) % 2)]
        self.pos += len(data)
        return np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0

    @property
    def seconds(self) -> float:
        return self.pos / 2 / RATE


class Journal:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.seq = 0
        if path.exists():
            for ln in path.read_text(encoding="utf-8", errors="replace").splitlines():
                try:
                    self.seq = max(self.seq, json.loads(ln).get("seq", 0))
                except Exception:
                    pass
        self.f = open(path, "a", encoding="utf-8")

    def add(self, p: Piece, model: str) -> None:
        self.seq += 1
        row = {"seq": self.seq, "track": p.track, "t_start": round(p.t_start, 2), "t_end": round(p.t_end, 2),
               "text": p.text, "lang": p.lang, "model": model, "committed_at": time.time()}
        if p.en:
            row["en"] = p.en                                    # English translation (Tamil pieces)
        if p.echo:
            row["echo"] = True
        if p.alt:
            row["alt"] = p.alt                                  # the other engine's reading (hint for LLM clean-up)
        self.f.write(json.dumps(row, ensure_ascii=False) + "\n")
        self.f.flush(); os.fsync(self.f.fileno())


def _set_low_priority() -> None:
    try:
        import ctypes
        ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(), 0x00004000)  # BELOW_NORMAL
    except Exception:
        pass


def run(meeting: Path, model_key: str, language: str, vocabulary: str, threads: int,
        trigger_s: float, max_wait_s: float, journal_name: str = "journal.jsonl", status_name: str = "worker.json") -> int:
    raw = meeting / "raw"
    status_path = meeting / status_name
    t_begin = time.time()
    mode = language if language in MODES else "hybrid"
    use_indic = mode in ("hybrid", "ta")
    use_whisper = mode in ("hybrid", "en", "whisper")

    def put(state: str, **kw) -> None:
        d = {"state": state, "t": time.time(), "pid": os.getpid(), "model": model_key, "mode": mode, **kw}
        tmp = status_path.with_suffix(".tmp")
        try:
            tmp.write_text(json.dumps(d), encoding="utf-8"); os.replace(tmp, status_path)
        except Exception:
            pass

    put("loading")
    tr = indic = None
    try:
        if use_whisper:
            tr = Transcriber(paths.MODELS_DIR / MODEL_DIRS[model_key], threads=threads)
        if use_indic:
            from .indic import IndicConformer
            indic = IndicConformer(paths.MODELS_DIR / "indic-asr", threads=threads)
    except Exception as e:
        put("error", error=f"could not load speech model: {e}")
        return 2
    model_label = "indic+" + tr.name if (tr and indic) else ("indic-conformer" if indic else tr.name)

    journal = Journal(meeting / journal_name)
    tails = {t: PcmTail(raw / f"{t}.pcm") for t in TRACKS}
    short = mode in ("hybrid", "ta")                  # Indic is trained on short clips: cap utterances at ~8.5 s
    segs = {t: TrackSegmenter(t, max_s=8.5, min_silence_s=0.5) if short else TrackSegmenter(t) for t in TRACKS}
    pending: Dict[str, List[Utterance]] = {t: [] for t in TRACKS}
    first_seen: Dict[str, float] = {}
    context: Dict[str, str] = {t: "" for t in TRACKS}
    done_pieces = speech_done = decode_time = 0.0
    last_put = 0.0
    finishing = False

    def pend_s(t: str) -> float:
        return sum(u.dur for u in pending[t])

    def commit(pieces: List[Piece]) -> None:
        nonlocal done_pieces
        for p in pieces:
            if p.track == "mic":
                try:
                    p.echo = echo.is_echo(raw, p.t_start, p.t_end)      # speaker bleed -> flagged, hidden by the reader
                except Exception:
                    pass
            journal.add(p, model_label); done_pieces += 1

    def indic_text(u: Utterance) -> str:
        try:
            return tamil.clean_tamil(indic.transcribe(u.audio, "ta"))
        except Exception:
            traceback.print_exc(file=sys.stderr)
            return ""

    def on_utterances(t: str, us: List[Utterance]) -> None:
        nonlocal speech_done, decode_time
        for u in us:
            if indic:
                t0 = time.time(); u.ta = indic_text(u); decode_time += time.time() - t0     # type: ignore[attr-defined]
            if mode == "ta":
                speech_done += u.dur
                if u.ta:                                                                  # type: ignore[attr-defined]
                    commit([Piece(t, u.start_s, u.end_s, u.ta, "ta")])                    # type: ignore[attr-defined]
            else:
                pending[t].append(u); first_seen.setdefault(t, time.time())

    def run_pack(t: str, force: bool) -> None:
        nonlocal speech_done, decode_time
        packs = tr.pack(pending[t])
        todo = packs if force else packs[:-1] + ([packs[-1]] if sum(u.dur for u in packs[-1]) >= trigger_s else [])
        if not todo:
            return
        used = sum(len(p) for p in todo)
        pending[t] = pending[t][used:]
        if not pending[t]:
            first_seen.pop(t, None)
        else:
            first_seen[t] = time.time()
        for pack in todo:
            t0 = time.time()
            pieces: List[Piece] = []
            try:
                if mode == "hybrid":
                    texts = tr.transcribe_utterances(pack, vocabulary, context[t])
                    for i, (u, en) in enumerate(zip(pack, texts)):          # Whisper sometimes skips an utterance inside a
                        ta = getattr(u, "ta", "")                           # mixed pack: re-decode that one on its own, or
                        if not en and len(ta.split()) >= 3 and u.dur >= 1.0:   # English would fall back to Tamil-script gibberish
                            texts[i] = tr.transcribe_utterances([u], vocabulary, "")[0]
                    for u, en in zip(pack, texts):
                        ta = getattr(u, "ta", "")
                        if not en and not ta:
                            continue
                        if en and (not ta or lid.is_english(en, ta)):
                            pieces.append(Piece(t, u.start_s, u.end_s, en, "en", alt=ta))
                        else:
                            pieces.append(Piece(t, u.start_s, u.end_s, ta, "ta", alt=en))
                    tam_idx = [i for i, pc in enumerate(pieces) if pc.lang == "ta"]
                    if tam_idx:                                     # English translation of the Tamil pieces (one extra packed decode)
                        by_start = {round(u.start_s, 2): u for u in pack}
                        tu = [by_start[round(pieces[i].t_start, 2)] for i in tam_idx]
                        try:
                            tr_texts = tr.translate_utterances(tu, vocabulary, "")
                        except Exception:
                            traceback.print_exc(file=sys.stderr); tr_texts = [""] * len(tu)
                        for i, e in zip(tam_idx, tr_texts):
                            pieces[i].en = e
                else:
                    pieces = tr.transcribe_pack(pack, "en" if mode == "en" else "mixed", vocabulary, context[t])
            except Exception:
                traceback.print_exc(file=sys.stderr)
            decode_time += time.time() - t0
            speech_done += sum(u.dur for u in pack)
            commit(pieces)
            if pieces:
                context[t] = " ".join(p.text for p in pieces if p.lang != "ta")

    put("listening")
    while True:
        got = False
        for t in TRACKS:
            x = tails[t].read()
            if len(x):
                got = True
                for k in range(0, len(x), RATE):                # ~1 s blocks: low VAD latency, efficient calls
                    us = segs[t].feed(x[k:k + RATE])
                    if us:
                        on_utterances(t, us)
        if not finishing and (raw / "END").exists() and not got:
            finishing = True
            for t in TRACKS:
                us = segs[t].flush()
                if us:
                    on_utterances(t, us)
            put("draining")
        now = time.time()
        if use_whisper:
            for t in TRACKS:
                if pending[t]:
                    age = now - first_seen.get(t, now)
                    if finishing or pend_s(t) >= trigger_s or age > max_wait_s:
                        run_pack(t, force=finishing or age > max_wait_s)
                        break                                    # re-poll the files between packs
        if finishing and not any(pending[t] for t in TRACKS):
            break
        if now - last_put > 0.5:
            last_put = now
            put("draining" if finishing else "listening",
                audio_s=max(tails[t].seconds for t in TRACKS), pending_s=sum(pend_s(t) for t in TRACKS),
                pieces=int(done_pieces), speech_done_s=round(speech_done, 1),
                rtf=round(decode_time / speech_done, 3) if speech_done else None)
        if not got:
            time.sleep(0.25)
    put("done", audio_s=max(tails[t].seconds for t in TRACKS), pieces=int(done_pieces),
        speech_done_s=round(speech_done, 1), rtf=round(decode_time / speech_done, 3) if speech_done else None,
        wall_s=round(time.time() - t_begin, 1))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--meeting-dir", required=True)
    ap.add_argument("--model", default="small", help="Whisper model for the English / Whisper part")
    ap.add_argument("--language", default="hybrid", choices=list(MODES) + ["mixed"])
    ap.add_argument("--vocabulary", default="")
    ap.add_argument("--threads", type=int, default=3)
    ap.add_argument("--journal", default="journal.jsonl")
    ap.add_argument("--status", default="worker.json")
    a = ap.parse_args()
    _set_low_priority()
    lang = "hybrid" if a.language == "mixed" else a.language      # 'mixed' = old name of the default
    trigger = {"large-v3-turbo": 22.0, "medium": 18.0, "small": 14.0}.get(a.model, 18.0)
    return run(Path(a.meeting_dir), a.model, lang, a.vocabulary, a.threads, trigger, max_wait_s=trigger + 12,
               journal_name=a.journal, status_name=a.status)


if __name__ == "__main__":
    sys.exit(main())
