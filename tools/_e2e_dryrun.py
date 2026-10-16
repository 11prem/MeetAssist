"""End-to-end worker -> journal -> export dry run (no audio devices).

Creates a synthetic meeting, seeds its queue with real test audio as if the
capture pipeline had produced segments, runs the Worker exactly as `_worker`
does, then exports. Prints the resulting txt. Not a pytest test in the suite
(needs a model download); run manually to smoke the 5.3 wiring.
"""

import os, sys, tempfile, wave
from pathlib import Path
import numpy as np

os.environ["MEETASSIST_DATA_DIR"] = tempfile.mkdtemp(prefix="meet_e2e_")
os.environ["MEETASSIST_MODEL"] = os.environ.get("MEETASSIST_MODEL", "ggml-small.bin")

from meetassist.asr.whispercpp import WhisperCppBackend
from meetassist.config import ASR_MODEL, MODELS_DIR
from meetassist.pipeline.worker import Worker
from meetassist.queue.diskqueue import DiskQueue
from meetassist.store import control
from meetassist.store.journal import Journal
from meetassist.export.writers import write_all

with wave.open("test_dl/silero-vad-6.0/tests/data/test.wav", "rb") as w:
    a = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768.0

mid = control.new_meeting_id()
d = control.meeting_dir(mid)
for sub in ("queue", "raw", "exports"):
    (d / sub).mkdir(parents=True, exist_ok=True)
control.write_meta(mid, {"id": mid, "name": "e2e", "model": ASR_MODEL})

q = DiskQueue(control.queue_dir(mid))
q.enqueue("mic", 5000, a[: 8 * 16000])
q.enqueue("loop", 5000, a[: 8 * 16000])

backend = WhisperCppBackend(n_threads=4)
backend.load(MODELS_DIR, model=ASR_MODEL)
w = Worker(backend, control.queue_dir(mid), control.journal_path(mid), model=ASR_MODEL)
while w.run_once():
    pass
w.finalize()
backend.unload()

recs = list(Journal.read(control.journal_path(mid)))
print(f"mid={mid} committed={len(recs)}")
paths = write_all(recs, control.exports_dir(mid), stem=mid, title="e2e",
                  date="2026-10-04", duration_s=8, model=ASR_MODEL)
print("--- txt ---")
print(paths[0].read_text(encoding="utf-8"))
