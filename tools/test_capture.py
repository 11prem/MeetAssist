"""1.4 validation: resampler unit check + short dual capture to WAVs.

Run: .venv\\Scripts\\python.exe tools/test_capture.py
Mic should produce a non-empty WAV (ambient noise at minimum). Loopback may be
empty if no audio is rendered on the endpoint (expected on an idle machine).
"""

from __future__ import annotations

import sys
import tempfile
import wave
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from meetassist.audio.capture import _Resampler, Recorder

RATE = 16000
DURATION = 3


def _test_resampler() -> None:
    r = _Resampler(48000, 16000)
    # 1 s of 1 kHz sine at 48k -> expect 16k samples, 1 kHz, contiguous blocks
    n = 48000
    t = np.arange(n) / 48000.0
    x = (np.sin(2 * np.pi * 1000 * t)).astype(np.float32)
    out: list[np.ndarray] = []
    for i in range(0, n, 4800):
        out.append(r.process(x[i : i + 4800]))
    y = np.concatenate(out)
    assert abs(len(y) - 16000) < 64, f"bad output length {len(y)}"
    assert abs(np.max(y) - 1.0) < 0.02, "peak lost"
    # rough dominant-frequency check via zero crossings
    zc = np.sum(np.diff(np.signbit(y)))
    freq = zc / 2 / (len(y) / RATE)
    assert 900 < freq < 1100, f"freq drift: {freq}"
    print(f"resampler OK: {len(y)} samples, ~{freq:.0f} Hz, peak {np.max(y):.3f}")


def _write_wav(path: Path, blocks: list[np.ndarray]) -> None:
    if not blocks:
        path.write_bytes(b"")  # empty file flagged below
        return
    data = np.concatenate(blocks)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(RATE)
        w.writeframes((data * 32767).astype(np.int16).tobytes())


def main() -> int:
    _test_resampler()

    mic_blocks: list[np.ndarray] = []
    loop_blocks: list[np.ndarray] = []
    rec = Recorder(mic_blocks.append, loop_blocks.append)
    rec.start()
    import time

    time.sleep(DURATION)
    rec.stop()

    tmp = Path(tempfile.gettempdir())
    mic_wav, loop_wav = tmp / "ma_mic.wav", tmp / "ma_loop.wav"
    _write_wav(mic_wav, mic_blocks)
    _write_wav(loop_wav, loop_blocks)
    print(f"mic:   {len(mic_blocks)} blocks -> {mic_wav} ({mic_wav.stat().st_size} B)")
    print(f"loop:  {len(loop_blocks)} blocks -> {loop_wav} ({loop_wav.stat().st_size} B)")
    ok = mic_wav.stat().st_size > 44  # at least an empty WAV header worth of data
    print("PASS: mic captured 16k mono" if ok else "FAIL: mic silent")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
