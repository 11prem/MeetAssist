"""Capture subprocess: mic + WASAPI loopback -> 16 kHz mono int16 raw files.

Runs as its own process (``python -m meetassist.capture_proc ...``) so that a PortAudio
crash or a device change can never take the GUI down or lose audio already on disk
(old v1 finding: a 2nd PyAudio instance per process hard-crashes; a fresh process is safe).

Contract with the GUI (all via files in <meeting>/raw/):
  mic.pcm, loop.pcm   append-only, int16 LE, 16 kHz mono. Sample N == N/16000 s after
                      the meeting epoch (``--epoch``), kept aligned to the wall clock by
                      zero-padding when a track delivers nothing (WASAPI loopback delivers
                      NO data while nothing is playing) — so both tracks share one timeline
                      and a respawn after a device change continues on the same timeline.
  status.json         rewritten ~2x/s: levels, device names, state, heartbeat.
  STOP                created by the GUI to request a clean stop.
Exit codes: 0 clean stop, 3 device changed (GUI should respawn), 2 fatal.
"""
from __future__ import annotations

import argparse
import json
import os
import queue
import sys
import threading
import time
from pathlib import Path

import numpy as np

RATE = 16000
PAD_THRESHOLD_S = 0.30      # tolerate this much jitter before zero-padding
SUPERVISE_S = 2.0


class _Track:
    def __init__(self, name: str, path: Path, epoch: float) -> None:
        self.name, self.epoch = name, epoch
        self.f = open(path, "ab", buffering=0)
        self.written = self.f.tell() // 2          # samples already on disk (respawn-safe)
        self.q: "queue.Queue[np.ndarray]" = queue.Queue()
        self.rms = 0.0
        self.last_data = 0.0
        self.dev = ""
        self.stream = None
        self.resampler = None
        self.channels = 1

    def expected(self, now: float) -> int:
        return int((now - self.epoch) * RATE)

    def pad_to(self, n: int) -> None:
        if n > self.written:
            self.f.write(b"\x00\x00" * (n - self.written))
            self.written = n

    def write(self, x: np.ndarray, now: float) -> None:
        lag = self.expected(now) - len(x) - self.written
        if lag > PAD_THRESHOLD_S * RATE:
            self.pad_to(self.written + lag)
        self.f.write(x.tobytes())
        self.written += len(x)


def _make_resampler(src_rate: int):
    import av
    return av.AudioResampler(format="s16", layout="mono", rate=RATE)


def _open_stream(pa, pyaudio, dev: dict, track: _Track):
    import av
    src_rate = int(dev["defaultSampleRate"])
    ch = max(1, int(dev["maxInputChannels"]))
    track.channels, track.dev = ch, dev["name"]
    track.resampler = _make_resampler(src_rate)

    def cb(in_data, frames, ti, status):
        try:
            a = np.frombuffer(in_data, dtype=np.float32)
            if ch > 1:
                a = a.reshape(-1, ch).mean(axis=1)
            frame = av.AudioFrame.from_ndarray(a.reshape(1, -1).astype(np.float32), format="flt", layout="mono")
            frame.sample_rate = src_rate
            out = b"".join(bytes(f.planes[0]) for f in track.resampler.resample(frame))
            if out:
                x = np.frombuffer(out, dtype=np.int16)
                track.rms = float(np.sqrt(np.mean((x.astype(np.float32) / 32768) ** 2)))
                track.last_data = time.time()
                track.q.put(x.copy())
        except Exception:
            pass
        return (None, pyaudio.paContinue)

    s = pa.open(format=pyaudio.paFloat32, channels=ch, rate=src_rate, input=True,
                input_device_index=dev["index"], frames_per_buffer=max(src_rate // 10, 1),
                stream_callback=cb)
    s.start_stream()
    track.stream = s


def _alive(pid: int) -> bool:
    """Process liveness on Windows (os.kill(pid, 0) would send Ctrl+C there!)."""
    try:
        import ctypes
        k = ctypes.windll.kernel32
        h = k.OpenProcess(0x1000, False, pid)           # PROCESS_QUERY_LIMITED_INFORMATION
        if not h:
            return False
        code = ctypes.c_ulong()
        k.GetExitCodeProcess(h, ctypes.byref(code)); k.CloseHandle(h)
        return code.value == 259                        # STILL_ACTIVE
    except Exception:
        return True


def _ident(d: dict | None):
    return None if not d else (d.get("index"), d.get("name"))


def list_mics() -> None:
    """Print a JSON list of microphone names (used by the Settings dialog)."""
    import pyaudiowpatch as pyaudio
    pa = pyaudio.PyAudio()
    names = []
    wasapi = next((i for i in range(pa.get_host_api_count()) if "WASAPI" in pa.get_host_api_info_by_index(i)["name"]), None)
    for i in range(pa.get_device_count()):
        d = pa.get_device_info_by_index(i)
        if d["maxInputChannels"] > 0 and not d.get("isLoopbackDevice") and d["hostApi"] == wasapi and d["name"] not in names:
            names.append(d["name"])
    pa.terminate()
    print(json.dumps(names))


def main() -> int:
    if "--list" in sys.argv:
        list_mics()
        return 0
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", required=True)
    ap.add_argument("--epoch", type=float, required=True, help="meeting start, unix seconds")
    ap.add_argument("--no-mic", action="store_true")
    ap.add_argument("--no-loop", action="store_true")
    ap.add_argument("--mic-name", default="", help="substring of preferred mic device name")
    a = ap.parse_args()
    raw = Path(a.raw_dir); raw.mkdir(parents=True, exist_ok=True)
    stop_file = raw / "STOP"
    status_file = raw / "status.json"
    parent = os.getppid()

    def put_status(state: str, **kw):
        d = {"state": state, "t": time.time(), "pid": os.getpid(), **kw}
        tmp = status_file.with_suffix(".tmp")
        try:
            tmp.write_text(json.dumps(d), encoding="utf-8"); os.replace(tmp, status_file)
        except Exception:
            pass

    try:
        import pyaudiowpatch as pyaudio
        pa = pyaudio.PyAudio()
    except Exception as e:
        put_status("error", error=f"audio init failed: {e}")
        return 2

    tracks: dict[str, _Track] = {}
    cur: dict[str, dict | None] = {"mic": None, "loop": None}

    def resolve_mic():
        if a.mic_name:
            for i in range(pa.get_device_count()):
                d = pa.get_device_info_by_index(i)
                if d["maxInputChannels"] > 0 and not d.get("isLoopbackDevice") and a.mic_name.lower() in d["name"].lower():
                    return d
        return pa.get_default_input_device_info()

    problems: list[str] = []
    if not a.no_mic:
        try:
            cur["mic"] = resolve_mic()
            t = tracks["mic"] = _Track("mic", raw / "mic.pcm", a.epoch)
            _open_stream(pa, pyaudio, cur["mic"], t)
        except Exception as e:
            tracks.pop("mic", None); cur["mic"] = None
            problems.append(f"microphone unavailable ({e}) - check Windows Settings > Privacy > Microphone")
    if not a.no_loop:
        try:
            cur["loop"] = pa.get_default_wasapi_loopback()
            t = tracks["loop"] = _Track("loop", raw / "loop.pcm", a.epoch)
            _open_stream(pa, pyaudio, cur["loop"], t)
        except Exception as e:
            tracks.pop("loop", None); cur["loop"] = None
            problems.append(f"system audio unavailable ({e})")
    if not tracks:
        put_status("error", error="Could not open any audio device. " + " | ".join(problems))
        return 2
    warn = " | ".join(problems) or None

    code, last_sup = 0, time.time()
    while True:
        now = time.time()
        for tr in tracks.values():
            drained = 0
            try:
                while True:
                    x = tr.q.get_nowait(); tr.write(x, now); drained += 1
            except queue.Empty:
                pass
            # idle endpoint (loopback silent): keep the timeline moving
            tr.pad_to(max(tr.written, tr.expected(now) - int(0.25 * RATE)))
        if stop_file.exists():
            break
        if not _alive(parent):
            break                     # GUI died: stop cleanly, audio is already on disk
        if now - last_sup > SUPERVISE_S:
            last_sup = now
            try:
                if "mic" in tracks and _ident(resolve_mic()) != _ident(cur["mic"]):
                    code = 3; break
                if "loop" in tracks and _ident(pa.get_default_wasapi_loopback()) != _ident(cur["loop"]):
                    code = 3; break
            except Exception:
                pass
        put_status("recording", warning=warn, epoch=a.epoch,
                   mic_rms=tracks["mic"].rms if "mic" in tracks else None,
                   loop_rms=tracks["loop"].rms if "loop" in tracks else None,
                   mic_dev=tracks["mic"].dev if "mic" in tracks else None,
                   loop_dev=tracks["loop"].dev if "loop" in tracks else None,
                   mic_age=(now - tracks["mic"].last_data) if "mic" in tracks and tracks["mic"].last_data else None,
                   loop_age=(now - tracks["loop"].last_data) if "loop" in tracks and tracks["loop"].last_data else None)
        time.sleep(0.1)

    # clean shutdown: stop streams first (join PortAudio thread), then flush
    for tr in tracks.values():
        try: tr.stream.stop_stream()
        except Exception: pass
    time.sleep(0.15)
    for tr in tracks.values():
        try:
            while True: tr.write(tr.q.get_nowait(), time.time())
        except queue.Empty:
            pass
        tr.pad_to(tr.expected(time.time()))
        tr.f.close()
        try: tr.stream.close()
        except Exception: pass
    put_status("stopped" if code == 0 else "device_changed")
    try: pa.terminate()
    except Exception: pass
    return code


if __name__ == "__main__":
    sys.exit(main())
