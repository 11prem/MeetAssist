"""GUI-independent control of meetings: spawn/stop capture + ASR worker processes, import files,
poll status. The GUI only calls this and renders what it returns.

Process model (see todo.md V2 log): capture_proc and asr_worker are separate OS processes that talk
through files in the meeting folder, so a crash in either never takes the GUI down or loses audio.
"""
from __future__ import annotations

import json
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Callable, List, Optional

from . import export, paths, store
from .settings import Settings

CREATE_NO_WINDOW = 0x08000000


def _python() -> str:
    exe = Path(sys.executable)
    cand = exe.with_name("python.exe")          # pythonw.exe has no stdio; children want python.exe + no window
    return str(cand if cand.exists() else exe)


def _spawn(module: str, args: List[str], log: Path) -> subprocess.Popen:
    env = dict(**__import__("os").environ)
    env["PYTHONIOENCODING"] = "utf-8"
    env["HF_HUB_OFFLINE"] = "1"                  # AGENTS rule 3: nothing may ever phone home
    env["TRANSFORMERS_OFFLINE"] = "1"
    env["PYTHONPATH"] = str(paths.APP_ROOT) + (";" + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    return subprocess.Popen([_python(), "-m", module, *args], cwd=str(paths.APP_ROOT), env=env,
                            stdin=subprocess.DEVNULL, stdout=open(log, "ab"), stderr=subprocess.STDOUT,
                            creationflags=CREATE_NO_WINDOW)


def _alive(p: Optional[subprocess.Popen]) -> bool:
    return p is not None and p.poll() is None


class Controller:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.active: Optional[Path] = None           # meeting being recorded right now
        self.capture: Optional[subprocess.Popen] = None
        self.worker: Optional[subprocess.Popen] = None
        self.respawns = 0
        self.import_progress: Optional[float] = None
        self._row_cache: dict = {}
        self.capture_failed = False
        self.refine_done: list = []                  # meetings whose refinement just finished (GUI drains this)
        self.refiners: dict = {}                     # meeting dir -> background 'refine' worker process
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ live recording
    def start_live(self, title: str = "") -> Path:
        if self.active:
            raise RuntimeError("a meeting is already being recorded")
        s = self.settings
        d = store.new_meeting(title, "live", s.model_name, s.language)
        self.active = d
        self.respawns = 0
        self.capture_failed = False
        self._spawn_worker(d)
        self._spawn_capture(d)
        return d

    def _spawn_capture(self, d: Path) -> None:
        s, meta = self.settings, store.read_meta(d)
        args = ["--raw-dir", str(d / "raw"), "--epoch", repr(float(meta["epoch"]))]
        if not s.record_mic: args.append("--no-mic")
        if not s.record_system_audio: args.append("--no-loop")
        if s.mic_name: args += ["--mic-name", s.mic_name]
        self.capture = _spawn("meetassist.capture_proc", args, d / "capture.log")

    def _spawn_worker(self, d: Path, model: Optional[str] = None, refine: bool = False) -> subprocess.Popen:
        s = self.settings
        args = ["--meeting-dir", str(d), "--model", model or s.model_name, "--language", s.language,
                "--threads", "3"]          # measured: 2..10 threads are equally fast; fewer = less CPU stolen from the call
        if refine:
            args += ["--journal", "journal.refine.jsonl", "--status", "worker.refine.json"]
        voc = s.vocabulary
        if voc: args += ["--vocabulary", voc]
        p = _spawn("meetassist.asr_worker", args, d / ("refine.log" if refine else "worker.log"))
        if refine:
            self.refiners[d] = p
        else:
            self.worker = p
        return p

    # ------------------------------------------------------------------ two-pass: background refinement
    def start_refine(self, d: Path, model: str) -> None:
        """Re-transcribe retained raw audio with a more accurate model into journal.refine.jsonl; the draft
        stays visible and untouched until the refined run completes, then they are swapped (draft kept)."""
        for f in ("journal.refine.jsonl", "worker.refine.json"):
            (d / f).unlink(missing_ok=True)
        store.update_meta(d, refine_model=model, refine_pending=True)
        self._spawn_worker(d, model, refine=True)

    def refine_state(self, d: Path) -> Optional[dict]:
        st = store.read_json(d / "worker.refine.json")
        return st

    def _finish_refine(self, d: Path) -> None:
        meta = store.read_meta(d)
        j, r = d / "journal.jsonl", d / "journal.refine.jsonl"
        if r.exists() and r.stat().st_size > 0:
            if j.exists():
                j.replace(d / "journal.draft.jsonl")       # AGENTS rule 4: the earlier text is never destroyed
            r.replace(j)
            (d / "cleaned.json").unlink(missing_ok=True)   # cleaned.json was keyed to the draft's seq numbers
            (d / "summary.md").rename(d / "summary.draft.md") if (d / "summary.md").exists() else None
            store.update_meta(d, model=meta.get("refine_model", meta.get("model")), refine_pending=False, refined=True)
            w = store.read_json(d / "worker.refine.json") or {}
            (d / "worker.json").write_text(json.dumps(w), encoding="utf-8")
        else:
            store.update_meta(d, refine_pending=False)
        self.refiners.pop(d, None)

    def tick(self) -> None:
        """Call ~1x/s from the GUI: respawn capture after a device change; complete finished refinements."""
        for d in list(self.refiners):
            st = store.read_json(d / "worker.refine.json") or {}
            if st.get("state") == "done":
                self._finish_refine(d)
                self.refine_done.append(d)
            elif self.refiners[d].poll() is not None and st.get("state") != "done":
                store.update_meta(d, refine_pending=False)          # refiner died: keep the draft, give up quietly
                self.refiners.pop(d, None)
        if not self.active:
            return
        if self.capture is not None and self.capture.poll() is not None and not (self.active / "raw" / "STOP").exists():
            code = self.capture.returncode
            if code == 2:                            # fatal (no device could be opened): do not loop forever
                self.capture_failed = True
            elif self.respawns < 20:                 # device change (3) or crash: carry on, same timeline
                self.respawns += 1
                time.sleep(0.5)
                self._spawn_capture(self.active)

    def stop_live(self, wait_s: float = 8.0) -> Path:
        d = self.active
        if not d:
            raise RuntimeError("no active meeting")
        (d / "raw" / "STOP").write_text("stop")
        t0 = time.time()
        while _alive(self.capture) and time.time() - t0 < wait_s:
            time.sleep(0.1)
        if _alive(self.capture):
            self.capture.kill()
        (d / "raw" / "END").write_text("end")      # tells the worker: no more audio -> flush and finish
        store.update_meta(d, ended_at=time.time(), duration=store.audio_seconds(d))
        self.active, self.capture = None, None
        return d

    def recover(self) -> List[Path]:
        """Meetings left open by a crash/power loss: finalise them so their worker can finish."""
        out = []
        for m in store.list_meetings():
            if m.meta.get("ended_at") is None and m.path != self.active:
                st = store.read_json(m.path / "raw" / "status.json")
                if st and time.time() - st.get("t", 0) < 5:
                    continue                          # still running somewhere
                (m.path / "raw" / "END").write_text("recovered")
                store.update_meta(m.path, ended_at=time.time(), duration=store.audio_seconds(m.path), recovered=True)
                if not (m.path / "worker.json").exists() or (store.read_json(m.path / "worker.json") or {}).get("state") != "done":
                    self._spawn_worker(m.path, m.meta.get("model") or None)
                out.append(m.path)
        return out

    # ------------------------------------------------------------------ import / re-run
    def import_file(self, src: str, title: str = "", on_done: Optional[Callable[[Path], None]] = None,
                    on_error: Optional[Callable[[str], None]] = None) -> Path:
        s = self.settings
        d = store.new_meeting(title or Path(src).stem, "import", s.model_name, s.language)
        store.update_meta(d, source=src)
        self.import_progress = 0.0
        self._spawn_worker(d)                        # worker tails the pcm while we are still decoding

        def run() -> None:
            try:
                secs = store.decode_to_pcm(src, d / "raw" / "mic.pcm", lambda f: setattr(self, "import_progress", f))
                store.update_meta(d, ended_at=time.time(), duration=secs)
            except Exception as e:
                store.update_meta(d, error=str(e))
                if on_error: on_error(str(e))
            finally:
                (d / "raw" / "END").write_text("end")
                self.import_progress = None
                if on_done: on_done(d)
        threading.Thread(target=run, daemon=True).start()
        return d

    def retranscribe(self, d: Path, model: Optional[str] = None) -> None:
        """Re-run ASR over the retained raw audio. The old journal is kept as journal.prev-<ts>.jsonl."""
        ts = time.strftime("%H%M%S")
        j = d / "journal.jsonl"
        if j.exists():
            j.rename(d / f"journal.prev-{ts}.jsonl")
        for f in ("worker.json", "cleaned.json"):
            (d / f).unlink(missing_ok=True)
        store.update_meta(d, model=model or self.settings.model_name)
        (d / "raw" / "END").write_text("end")
        self._spawn_worker(d, model)

    # ------------------------------------------------------------------ polling
    def _rows(self, d: Path) -> List[dict]:
        """Journal rows, re-parsed only when the file changed (the GUI polls ~2x/s)."""
        j = d / "journal.jsonl"
        try:
            stt = j.stat(); key = (stt.st_mtime_ns, stt.st_size)
        except FileNotFoundError:
            return []
        c = self._row_cache.get(d)
        if c and c[0] == key:
            return c[1]
        rows = export.read_journal(j)
        self._row_cache[d] = (key, rows)
        return rows

    def poll(self, d: Path) -> dict:
        w = store.read_json(d / "worker.json") or {}
        c = store.read_json(d / "raw" / "status.json") or {}
        stale = (time.time() - c.get("t", 0)) > 4 if c else True
        return {
            "worker": w, "capture": c, "capture_stale": stale,
            "rows": self._rows(d),
            "audio_s": store.audio_seconds(d),
            "recording": self.active == d,
        }
