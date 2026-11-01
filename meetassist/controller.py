"""GUI-independent control of meetings: spawn/stop capture + ASR worker processes, import files,
poll status. The GUI only calls this and renders what it returns.

Process model (see todo.md V2 log): capture_proc and asr_worker are separate OS processes that talk
through files in the meeting folder, so a crash in either never takes the GUI down or loses audio.
"""
from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
import traceback
from pathlib import Path
from typing import Callable, List, Optional

import numpy as np

from . import capture_proc, paths, store
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


def _pid_is_worker(pid: Optional[int], t: float) -> bool:
    """Live pid that was created before worker.json's last write `t` (after a crash/reboot a stale pid
    may belong to some unrelated, newer process)."""
    if not (pid and capture_proc._alive(int(pid))):
        return False
    try:
        import ctypes
        k = ctypes.windll.kernel32
        h = k.OpenProcess(0x1000, False, int(pid))              # PROCESS_QUERY_LIMITED_INFORMATION
        if not h:
            return False
        ft = [ctypes.c_ulonglong() for _ in range(4)]          # creation, exit, kernel, user (FILETIME)
        ok = k.GetProcessTimes(h, *[ctypes.byref(x) for x in ft]); k.CloseHandle(h)
        return not ok or (ft[0].value - 116444736000000000) / 1e7 <= t      # FILETIME (1601, 100 ns) -> unix s
    except Exception:
        return True


class Controller:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.active: Optional[Path] = None           # meeting being recorded right now
        self.capture: Optional[subprocess.Popen] = None
        self.worker: Optional[subprocess.Popen] = None
        self.worker_dir: Optional[Path] = None       # meeting that self.worker transcribes
        self.worker_respawns: dict = {}              # meeting dir -> times its dead worker was respawned
        self.respawns = 0
        self.import_progress: Optional[float] = None
        self._row_cache: dict = {}
        self.capture_failed = False
        self.refine_done: list = []                  # meetings whose refinement just finished (GUI drains this)
        self.refiners: dict = {}                     # meeting dir -> background 'refine' worker process
        self._voiceid_running: set = set()
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
            self.worker, self.worker_dir = p, d
        return p

    @staticmethod
    def _orphaned(d: Path, final: tuple = ("done", "error")) -> bool:
        """worker.json says unfinished and its recorded pid is gone. Never judged by staleness of `t`:
        a live worker does not update it during a long decode."""
        w = store.read_json(d / "worker.json") or {}
        return w.get("state") not in final and not _pid_is_worker(w.get("pid"), w.get("t", 0))

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
            for f in ("cleaned.json", "speakers.json"):    # both keyed to the draft's seq numbers
                (d / f).unlink(missing_ok=True)
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
        w, d = self.worker, self.worker_dir
        if w is not None and d is not None and w.poll() is not None:
            st = (store.read_json(d / "worker.json") or {}).get("state")
            if st not in ("done", "error") and self.worker_respawns.get(d, 0) < 3:   # cap: no crash loop
                self.worker_respawns[d] = self.worker_respawns.get(d, 0) + 1
                self._spawn_worker(d, store.read_meta(d).get("model") or None)       # resumes from the journal
            else:
                self.worker = self.worker_dir = None
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

    # ------------------------------------------------------------------ speaker voice recognition
    def identify_voices(self, d: Path) -> None:
        """Background pass (runs once per meeting, after transcription is done): embeds mic- and loopback-track
        speech and matches/clusters it against saved voice profiles. Never blocks the GUI."""
        if d in self._voiceid_running or (d / "speakers.json").exists():   # already done: re-running re-adds the same audio to centroids
            return
        self._voiceid_running.add(d)

        def run() -> None:
            from . import voiceid
            try:
                if voiceid.identify_speakers(d) is None:     # journal was swapped (refine) mid-run: redo for the new one
                    voiceid.identify_speakers(d)
            except Exception:
                traceback.print_exc()                    # stderr is app.log in the windowed app
            self._row_cache.pop(d, None)                 # force a re-read so names show up immediately
            self._voiceid_running.discard(d)
        threading.Thread(target=run, daemon=True).start()

    def record_voice_sample(self, seconds: float, ready_timeout: float = 20.0, stop_timeout: float = 5.0) -> np.ndarray:
        """Blocking: captures `seconds` of mic audio to a scratch folder (self-voice enrollment)
        and returns it as float32 mono 16 kHz. Call this off the GUI thread. Reuses the same
        capture subprocess as live recording rather than opening the microphone a second way.
        The `seconds` clock starts only once capture reports `recording` (interpreter + PortAudio startup
        can take seconds); raises RuntimeError with capture's own message if it fails to start."""
        tmp = paths.DATA_DIR / "_voice_sample"
        shutil.rmtree(tmp, ignore_errors=True)
        tmp.mkdir(parents=True, exist_ok=True)
        args = ["--raw-dir", str(tmp), "--epoch", repr(time.time()), "--no-loop"]
        if self.settings.mic_name:
            args += ["--mic-name", self.settings.mic_name]
        p = _spawn("meetassist.capture_proc", args, tmp / "capture.log")
        try:
            t0 = time.time()
            while True:
                st = store.read_json(tmp / "status.json") or {}
                if st.get("state") == "error" or p.poll() is not None:
                    raise RuntimeError(st.get("error") or f"microphone capture exited unexpectedly (code {p.poll()})")
                if st.get("state") == "recording":
                    break
                if time.time() - t0 > ready_timeout:
                    raise RuntimeError(f"microphone did not start within {ready_timeout:g} s")
                time.sleep(0.05)
            time.sleep(seconds)
            (tmp / "STOP").write_text("stop")
            try:
                p.wait(timeout=stop_timeout)
            except subprocess.TimeoutExpired:
                p.kill(); p.wait()                       # must be dead before its files are read/deleted (Windows keeps them locked)
            pcm = tmp / "mic.pcm"
            audio = (np.frombuffer(pcm.read_bytes(), dtype=np.int16).astype(np.float32) / 32768.0
                     if pcm.exists() else np.zeros(0, np.float32))
        finally:
            if p.poll() is None:
                p.kill(); p.wait()
            shutil.rmtree(tmp, ignore_errors=True)
        nz = np.flatnonzero(audio)                       # capture zero-pads from --epoch: drop startup/shutdown padding
        return audio[nz[0]:nz[-1] + 1] if nz.size else audio[:0]

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
                if self._orphaned(m.path, ("done",)):
                    self._spawn_worker(m.path, m.meta.get("model") or None)
                out.append(m.path)
            elif m.meta.get("ended_at") is not None and (m.path / "raw" / "END").exists() and self._orphaned(m.path):
                self._spawn_worker(m.path, m.meta.get("model") or None)      # worker died mid-meeting: resume it
                out.append(m.path)
        return out

    def delete_meeting(self, d: Path) -> bool:
        """Explicit user delete (AGENTS rule 2): stop the meeting's workers first - a running one keeps files
        open (half-deleted folder) and its models loaded. Returns whether the folder is gone (if not, the
        meeting stays listed and the delete can be retried)."""
        for p in (self.refiners.pop(d, None), self.worker if self.worker_dir == d else None):
            if _alive(p):                                   # tracked child: may not have written its status yet
                p.terminate()
                try:
                    p.wait(5)
                except subprocess.TimeoutExpired:
                    pass
        if self.worker_dir == d:
            self.worker = self.worker_dir = None
        for f in ("worker.json", "worker.refine.json"):     # untracked (e.g. started by an earlier app session)
            w = store.read_json(d / f) or {}
            pid = w.get("pid")
            if w.get("state") not in ("done", "error") and _pid_is_worker(pid, w.get("t", 0)):
                try:
                    os.kill(int(pid), signal.SIGTERM)                    # TerminateProcess on Windows
                except OSError:
                    pass
                t0 = time.time()
                while capture_proc._alive(int(pid)) and time.time() - t0 < 5:
                    time.sleep(0.05)
        t0 = time.time()
        while d in self._voiceid_running and time.time() - t0 < 5:     # its memmap keeps raw/*.pcm undeletable
            time.sleep(0.05)
        return store.delete_meeting(d)

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
        for f in ("worker.json", "cleaned.json", "speakers.json"):    # new journal restarts seq at 1
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
        rows = store.read_rows(d)
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
