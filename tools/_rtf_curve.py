import sys, time, wave
import numpy as np
from meetassist.asr.whispercpp import WhisperCppBackend
from meetassist.config import MODELS_DIR
SR = 16000
with wave.open("test_dl/silero-vad-6.0/tests/data/test.wav", "rb") as w:
    a = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768.0
model = sys.argv[1]
b = WhisperCppBackend(n_threads=4)
b.load(MODELS_DIR, model=model)
for sec in map(int, sys.argv[2:]):
    x = a[: sec * SR]
    t = time.perf_counter(); b.transcribe(x)
    el = time.perf_counter() - t
    print(f"seg {sec:>3}s: {el:>7.1f}s  RTF {el/sec:>6.2f}", flush=True)
b.unload()
