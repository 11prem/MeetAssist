"""Standalone capture diagnostic.

Enumerates WASAPI endpoints, then records ~10 s from the mic and the loopback,
printing live signal levels for each track. Run with:

    .venv\\Scripts\\python.exe tools/probe_devices.py

Exits 0 on success. Key finding this tool verifies: WASAPI loopback only
produces data while audio is *actually rendered* on the endpoint — on an idle
endpoint both blocking read() and callbacks stall. So loopback capture below
uses a timeout guard and reports "idle" rather than hanging. Play any audio to
exercise loopback.
"""

from __future__ import annotations

import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pyaudiowpatch as pyaudio

from meetassist.audio import devices

FORMAT = pyaudio.paFloat32
CHUNK_MS = 100
DURATION_S = 10
IDLE_TIMEOUT_S = 5  # loopback: stop (and warn) after this long with no data


def _rms(frame: bytes) -> float:
    import array

    return sum(x * x for x in array.array("f", frame)) / (len(frame) // 4)


def _capture_levels(pa: pyaudio.PyAudio, device: dict, label: str) -> float:
    # WASAPI shared mode only accepts each device's native rate/channel count
    # (loopback 48k/2ch, mic 44.1k). 16 kHz resampling is capture.py's job;
    # RMS is rate- and channel-independent so the probe records at native.
    rate = int(device["defaultSampleRate"])
    channels = int(device["maxInputChannels"])
    stream = pa.open(
        format=FORMAT,
        channels=channels,
        rate=rate,
        input=True,
        input_device_index=device["index"],
        frames_per_buffer=int(rate * CHUNK_MS / 1000),
    )

    buffers: list = []
    running = True
    before = time.time()

    def _run() -> None:
        try:
            for _ in range(DURATION_S * 1000 // CHUNK_MS):
                if not running:
                    return
                data = stream.read(stream._frames_per_buffer, exception_on_overflow=False)
                buffers.append(data)
        except Exception:
            pass

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()
    try:
        # If the loopback endpoint is idle, blocking read() never returns;
        # guard with a wall-clock timeout so we report it instead of hanging.
        deadline = time.time() + IDLE_TIMEOUT_S
        while thread.is_alive():
            if buffers:
                deadline = time.time() + IDLE_TIMEOUT_S  # got data -> keep going
            if time.time() > deadline and not buffers:
                break
            thread.join(timeout=0.2)
    finally:
        running = False
        stream.stop_stream()
        stream.close()

    peak = 0.0
    print(f"[{label}] recording {DURATION_S}s @ {rate} Hz ...")
    if not buffers:
        print(f"[{label}]  IDLE: no audio rendered on this endpoint "
              f"(loopback needs active playback). No levels captured.")
        return 0.0
    for data in buffers:
        rms = _rms(data)
        peak = max(peak, rms)
        print(f"[{label}] rms={rms:.4f} {'#' * min(int(rms * 400), 40)}")
    return peak


def main() -> int:
    pa = pyaudio.PyAudio()
    try:
        print("== WASAPI render endpoints ==")
        for e in devices.list_endpoints(pa):
            print(f"  {e['name']}  -> loopback idx {e['loopback_index']}")

        loop = devices.resolve_loopback(pa)
        mic = devices.resolve_mic(pa)
        print(f"\nmic:      {mic['name']}  (idx {mic['index']})")
        print(f"loopback: {loop['name']}  (idx {loop['index']})")

        mic_peak = _capture_levels(pa, mic, "mic")
        loop_peak = _capture_levels(pa, loop, "loop")
        print(f"\npeaks: mic={mic_peak:.4f} loop={loop_peak:.4f}")

        if mic_peak < 0.001:
            print("WARNING: mic is silent — is it muted / is anyone near it?")
            return 1
        if loop_peak < 0.001:
            print("NOTE: loopback idle (no audio rendered) — play something to test it.")
        return 0
    finally:
        pa.terminate()


if __name__ == "__main__":
    sys.exit(main())
