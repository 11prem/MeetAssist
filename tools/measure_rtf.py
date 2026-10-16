"""RTF measurement harness (task 4.3).

Measures real-time factor (transcription seconds / audio seconds) on this
laptop for both agreement modes, so we can decide whether the CPU sustains the
re-decode (plan §5.6 gate). Usage:

    python tools/measure_rtf.py [--model small|large] [--seconds 12]

Writes a small report to stdout. Not a test — this is a measurement for the
4.3 decision note.
"""

from __future__ import annotations

import argparse
import json
import tempfile
import time
import wave
from pathlib import Path

import numpy as np

from meetassist.asr.whispercpp import WhisperCppBackend
from meetassist.config import AgreementMode, MODELS_DIR
from meetassist.pipeline.worker import Worker
from meetassist.queue.diskqueue import DiskQueue

SR = 16000
MODELS = {"small": "ggml-small.bin", "large": "ggml-large-v3-turbo.bin"}


def _audio(seconds: int) -> np.ndarray:
    with wave.open("test_dl/silero-vad-6.0/tests/data/test.wav", "rb") as w:
        a = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768.0
    n = min(int(seconds * SR), len(a))
    return a[:n]


def measure(model: str, audio: np.ndarray, mode: AgreementMode, n_threads: int) -> dict:
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        q = DiskQueue(d / "q")
        # feed audio in several adjacent 6 s segments to create joins
        for i, cut in enumerate(range(0, len(audio), 6 * SR)):
            q.enqueue("mic", (cut // SR) * 1000 + 5000, audio[cut : cut + 6 * SR])
        backend = WhisperCppBackend(n_threads=n_threads)
        backend.load(MODELS_DIR, model=MODELS[model])
        w = Worker(backend, d / "q", d / "o.jsonl", agreement_mode=mode)
        t0 = time.perf_counter()
        while w.run_once():
            pass
        w.finalize()
        elapsed = time.perf_counter() - t0
        backend.unload()
    aud_s = len(audio) / SR
    return {
        "model": model,
        "mode": mode.value,
        "audio_s": round(aud_s, 1),
        "elapsed_s": round(elapsed, 1),
        "rtf": round(elapsed / aud_s, 2),
        "n_threads": n_threads,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="large", choices=MODELS.keys())
    ap.add_argument("--seconds", type=int, default=12)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--mode", default="both", choices=["off", "on", "both"])
    a = ap.parse_args()
    audio = _audio(a.seconds)
    print(f"audio: {a.seconds}s  model: {a.model}  threads: {a.threads}", flush=True)
    run = lambda m: print(f"  {m.value}: elapsed {measure(a.model, audio, m, a.threads)['elapsed_s']}s  RTF {measure(a.model, audio, m, a.threads)['rtf']}  ({a.seconds}s audio)", flush=True)
    print("running OFF...", flush=True); run(AgreementMode.OFF)
    if a.mode in ("on", "both"):
        print("running ON...", flush=True); run(AgreementMode.ON)


if __name__ == "__main__":
    main()
