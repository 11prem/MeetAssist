import numpy as np
from meetassist import echo

def _write(tmp_path, mic, loop):
    (mic * 32767).astype(np.int16).tofile(tmp_path / "mic.pcm"); (loop * 32767).astype(np.int16).tofile(tmp_path / "loop.pcm")

def _sig(f, mod, ph=0.0, n=6):
    t = np.arange(16000 * n) / 16000
    return (np.sin(2 * np.pi * f * t) * (0.3 + 0.3 * np.sin(2 * np.pi * mod * t + ph))).astype(np.float32)

def test_bleed_is_detected_even_when_delayed(tmp_path):
    loop = _sig(200, 1.3); mic = 0.5 * np.roll(loop, 300)           # same envelope, small acoustic delay
    _write(tmp_path, mic, loop); assert echo.is_echo(tmp_path, 0, 6)

def test_quiet_mic_under_loud_remote_is_bleed(tmp_path):
    loop = _sig(200, 1.3); mic = 0.05 * _sig(150, 0.7, 1.0)          # uncorrelated but 20x quieter than the speakers
    _write(tmp_path, mic, loop); assert echo.is_echo(tmp_path, 0, 6)

def test_user_talking_over_remote_is_kept(tmp_path):
    loop = _sig(200, 1.3) * 0.4; mic = _sig(150, 0.7, 1.0)           # loud, different envelope
    _write(tmp_path, mic, loop); assert not echo.is_echo(tmp_path, 0, 6)

def test_silent_loopback_is_never_echo(tmp_path):
    _write(tmp_path, _sig(150, 0.7), np.zeros(16000 * 6, np.float32)); assert not echo.is_echo(tmp_path, 0, 6)
