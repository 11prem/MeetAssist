"""Speaker-bleed (echo) detection at audio level.

With laptop speakers the microphone re-records the remote participants. The mic's loudness envelope over such
an utterance follows the loopback track's envelope almost exactly (correlation ~0.8-0.95), whereas the user's
own voice does not. We compare 50 ms RMS envelopes of both raw tracks over the utterance window.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

RATE = 16000
FR = 800                          # 50 ms


def _slice(path: Path, t0: float, t1: float) -> np.ndarray:
    n = int((t1 - t0) * RATE)
    try:
        x = np.fromfile(path, dtype=np.int16, count=n, offset=max(int(t0 * RATE), 0) * 2)
    except (OSError, ValueError):
        return np.zeros(0, np.float32)
    return x.astype(np.float32) / 32768.0


def _env(x: np.ndarray) -> np.ndarray:
    k = len(x) // FR
    return np.sqrt((x[: k * FR].reshape(k, FR) ** 2).mean(axis=1)) if k else np.zeros(0)


def is_echo(raw_dir: Path, t0: float, t1: float, min_corr: float = 0.6) -> bool:
    mic, loop = _slice(raw_dir / "mic.pcm", t0, t1), _slice(raw_dir / "loop.pcm", t0, t1)
    n = min(len(mic), len(loop))
    if n < FR * 6:
        return False
    em, el = _env(mic[:n]), _env(loop[:n])
    if (el > 0.002).mean() < 0.5:                    # loopback mostly silent -> nothing to bleed from
        return False
    rm, rl = float(np.sqrt((mic[:n] ** 2).mean())), float(np.sqrt((loop[:n] ** 2).mean()))
    if rm < 0.3 * rl:                                # mic far quieter than what the speakers play: it is bleed,
        return True                                  # a person talking into the mic is normally louder
    if em.std() < 1e-7 or el.std() < 1e-7:
        return False
    # allow a small acoustic delay between loopback and what the mic hears (+-100 ms)
    best = max(np.corrcoef(np.roll(em, s)[3:-3], el[3:-3])[0, 1] for s in (-2, -1, 0, 1, 2))
    return bool(best >= min_corr)
