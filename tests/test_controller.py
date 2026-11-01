import json, os, subprocess, sys, threading, time
from pathlib import Path

import numpy as np
import pytest

from meetassist import controller, paths, store, voiceid
from meetassist.controller import Controller
from meetassist.settings import Settings


def test_voiceid_failure_is_logged_not_swallowed(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(paths, "DATA_DIR", tmp_path)
    def boom(d):
        raise RuntimeError("embedder exploded")
    monkeypatch.setattr(voiceid, "identify_speakers", boom)
    c = Controller(Settings())
    c.identify_voices(tmp_path)
    end = time.time() + 5
    while c._voiceid_running and time.time() < end:
        time.sleep(0.01)
    assert not c._voiceid_running
    err = capsys.readouterr().err
    assert "embedder exploded" in err and "Traceback" in err


class _Proc:
    def __init__(self, code=None): self.code = code
    def poll(self): return self.code


def _meeting(tmp_path, monkeypatch, state, pid=999999, ended=True, t=None):
    monkeypatch.setattr(paths, "MEETINGS_DIR", tmp_path / "m")
    d = store.new_meeting("w", "live", "small")
    if ended:
        (d / "raw" / "END").write_text("end"); store.update_meta(d, ended_at=time.time())
    (d / "worker.json").write_text(json.dumps({"state": state, "pid": pid, "t": time.time() if t is None else t}), encoding="utf-8")
    return d


def _sleeper(d, name="worker.log"):
    log = open(d / name, "ab")                           # the child holds a file in the folder, like a real worker
    p = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"], stdout=log, stderr=subprocess.STDOUT)
    log.close()
    return p


def _reap(*ps):
    for p in ps:
        if p.poll() is None:
            p.kill()
        p.wait()


def _recording_spawn(monkeypatch, proc=lambda: _Proc(None)):
    calls = []
    def fake(module, args, log):
        calls.append(args); return proc()
    monkeypatch.setattr(controller, "_spawn", fake)
    return calls


def test_tick_respawns_dead_worker_until_cap(tmp_path, monkeypatch):
    d = _meeting(tmp_path, monkeypatch, "listening")
    calls = _recording_spawn(monkeypatch, lambda: _Proc(1))      # every worker dies straight away
    c = Controller(Settings())
    c._spawn_worker(d)
    for _ in range(6):
        c.tick()
    assert len(calls) == 1 + 3 and all(a[a.index("--meeting-dir") + 1] == str(d) for a in calls)


def test_tick_leaves_finished_worker_alone(tmp_path, monkeypatch):
    d = _meeting(tmp_path, monkeypatch, "done")
    calls = _recording_spawn(monkeypatch, lambda: _Proc(0))
    c = Controller(Settings())
    c._spawn_worker(d); c.tick()
    assert len(calls) == 1


def test_recover_respawns_orphaned_worker_of_ended_meeting(tmp_path, monkeypatch):
    d = _meeting(tmp_path, monkeypatch, "listening")
    calls = _recording_spawn(monkeypatch)
    Controller(Settings()).recover()
    assert len(calls) == 1 and str(d) in calls[0]


def test_recover_skips_done_and_live_workers(tmp_path, monkeypatch):
    _meeting(tmp_path, monkeypatch, "done")
    _meeting(tmp_path, monkeypatch, "listening", pid=os.getpid())   # still busy (worker.json is stale mid-decode)
    open_ = _meeting(tmp_path, monkeypatch, "listening", pid=os.getpid(), ended=False)   # GUI crashed, worker lives on
    calls = _recording_spawn(monkeypatch)
    Controller(Settings()).recover()
    assert calls == [] and (open_ / "raw" / "END").exists()          # the live worker finishes on its own


def test_recover_respawns_when_recorded_pid_was_reused(tmp_path, monkeypatch):
    d = _meeting(tmp_path, monkeypatch, "listening", pid=os.getpid(), t=time.time() - 10**6)   # pid now = a newer process
    calls = _recording_spawn(monkeypatch)
    Controller(Settings()).recover()
    assert len(calls) == 1 and str(d) in calls[0]


def test_pid_started_after_last_status_write_is_not_the_worker():
    assert controller._pid_is_worker(os.getpid(), time.time())
    assert not controller._pid_is_worker(os.getpid(), time.time() - 10**6)
    assert not controller._pid_is_worker(999999, time.time()) and not controller._pid_is_worker(None, time.time())


def test_delete_meeting_stops_an_untracked_worker_by_pid(tmp_path, monkeypatch):
    d = _meeting(tmp_path, monkeypatch, "listening")
    p = _sleeper(d)                                      # e.g. started by recover() in an earlier app session
    try:
        (d / "worker.json").write_text(json.dumps({"state": "listening", "pid": p.pid, "t": time.time()}), encoding="utf-8")
        assert Controller(Settings()).delete_meeting(d) is True
        assert p.poll() is not None and not d.exists()
    finally:
        _reap(p)


def test_delete_meeting_does_not_kill_a_reused_pid(tmp_path, monkeypatch):
    d = _meeting(tmp_path, monkeypatch, "listening")
    p = _sleeper(tmp_path, "other.log")                  # unrelated process that got the crashed worker's pid
    try:
        (d / "worker.json").write_text(json.dumps({"state": "listening", "pid": p.pid, "t": time.time() - 10**6}), encoding="utf-8")
        assert Controller(Settings()).delete_meeting(d) is True
        assert p.poll() is None and not d.exists()
    finally:
        _reap(p)


def test_delete_meeting_stops_tracked_workers_that_have_not_written_status(tmp_path, monkeypatch):
    d = _meeting(tmp_path, monkeypatch, "listening")     # stale worker.json of a dead predecessor (pid 999999)
    w, r = _sleeper(d), _sleeper(d, "refine.log")        # just spawned: no status of their own yet
    try:
        c = Controller(Settings())
        c.worker, c.worker_dir, c.refiners[d] = w, d, r
        assert c.delete_meeting(d) is True
        assert w.poll() is not None and r.poll() is not None and not d.exists()
        c.tick(); assert c.worker is None and d not in c.refiners    # a deliberately stopped worker is not respawned
    finally:
        _reap(w, r)


def test_finish_refine_drops_stale_speakers_json(tmp_path, monkeypatch):
    """#1: the refined journal restarts seq at 1, so the old seq->voice map would label the wrong lines."""
    d = _meeting(tmp_path, monkeypatch, "done")
    (d / "journal.jsonl").write_text("{}\n", encoding="utf-8"); (d / "journal.refine.jsonl").write_text("{}\n", encoding="utf-8")
    store.write_speakers(d, {1: "p1"})
    Controller(Settings())._finish_refine(d)
    assert not (d / "speakers.json").exists()


def test_retranscribe_drops_stale_speakers_json(tmp_path, monkeypatch):
    d = _meeting(tmp_path, monkeypatch, "done")
    (d / "journal.jsonl").write_text("{}\n", encoding="utf-8"); store.write_speakers(d, {1: "p1"})
    c = Controller(Settings()); monkeypatch.setattr(c, "_spawn_worker", lambda *a, **k: None)
    c.retranscribe(d)
    assert not (d / "speakers.json").exists()


def test_identify_voices_skips_a_meeting_that_already_has_speakers_json(tmp_path, monkeypatch):
    """#5: re-running on every launch re-added the same audio into the profile centroids."""
    calls = []
    monkeypatch.setattr(voiceid, "identify_speakers", lambda d: calls.append(d))
    store.write_speakers(tmp_path, {1: "p1"})
    c = Controller(Settings())
    c.identify_voices(tmp_path)
    assert not c._voiceid_running
    time.sleep(0.2)                                  # a (wrongly) started thread would have run by now
    assert calls == []


def test_identify_voices_reruns_once_when_the_journal_changed_mid_run(tmp_path, monkeypatch):
    results = iter([None, ["p1"]])                    # first pass went stale (refine swapped the journal), second is current
    calls = []
    monkeypatch.setattr(voiceid, "identify_speakers", lambda d: calls.append(d) or next(results))
    c = Controller(Settings())
    c.identify_voices(tmp_path)
    end = time.time() + 5
    while c._voiceid_running and time.time() < end:
        time.sleep(0.01)
    assert not c._voiceid_running and calls == [tmp_path, tmp_path]


class _Cap:
    """Fake capture_proc: after `delay` writes status.json (+ mic.pcm), then runs until STOP appears."""
    def __init__(self, raw, state="recording", delay=0.15, pcm=None, error="boom", hang=False):
        self.raw, self.code, self.hang, self.killed, self.t_status, self.t_stop = Path(raw), None, hang, False, None, None
        self.events = []
        self.kw = dict(state=state, delay=delay, pcm=pcm, error=error)
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self):
        k = self.kw; time.sleep(k["delay"])
        if k["state"] == "error":
            (self.raw / "status.json").write_text(json.dumps({"state": "error", "error": k["error"]})); self.code = 2; return
        if k["state"] == "never":
            return
        if k["pcm"] is not None:
            (self.raw / "mic.pcm").write_bytes(k["pcm"])
        (self.raw / "status.json").write_text(json.dumps({"state": "recording"})); self.t_status = time.time()
        while not (self.raw / "STOP").exists():
            time.sleep(0.005)
        self.t_stop = time.time()
        if not self.hang:
            self.code = 0

    def poll(self): return self.code
    def kill(self): self.killed = True; self.events.append("kill"); self.code = -9
    def wait(self, timeout=None):
        self.events.append("wait")
        end = time.time() + (timeout if timeout is not None else 5)
        while self.code is None and time.time() < end:
            time.sleep(0.005)
        if self.code is None:
            raise subprocess.TimeoutExpired("cap", timeout)
        return self.code


def _voice_ctrl(tmp_path, monkeypatch, **cap_kw):
    monkeypatch.setattr(paths, "DATA_DIR", tmp_path)
    made = []
    def fake(module, args, log):
        made.append(_Cap(args[args.index("--raw-dir") + 1], **cap_kw)); return made[-1]
    monkeypatch.setattr(controller, "_spawn", fake)
    return Controller(Settings()), made


def test_record_voice_sample_times_from_recording_and_trims_padding(tmp_path, monkeypatch):
    """Startup time becomes leading zero samples (capture pads from --epoch); the timer must start only once capture is recording."""
    tone = (np.sin(np.arange(1, 1601) / 5) * 8000).astype(np.int16)
    pcm = np.concatenate([np.zeros(800, np.int16), tone, np.zeros(500, np.int16)]).tobytes()
    c, made = _voice_ctrl(tmp_path, monkeypatch, pcm=pcm, delay=0.2)
    audio = c.record_voice_sample(0.3)
    assert len(audio) == 1600 and audio[0] != 0 and audio[-1] != 0
    assert made[0].t_stop - made[0].t_status >= 0.3 - 0.02          # STOP came `seconds` after "recording" appeared, not after spawn
    assert not (tmp_path / "_voice_sample").exists()


def test_record_voice_sample_raises_with_the_capture_error(tmp_path, monkeypatch):
    c, made = _voice_ctrl(tmp_path, monkeypatch, state="error", error="microphone unavailable (denied)")
    with pytest.raises(RuntimeError, match="microphone unavailable"):
        c.record_voice_sample(0.1)
    assert not (tmp_path / "_voice_sample").exists()


def test_record_voice_sample_gives_up_if_capture_never_starts(tmp_path, monkeypatch):
    c, made = _voice_ctrl(tmp_path, monkeypatch, state="never")
    with pytest.raises(RuntimeError, match="did not start"):
        c.record_voice_sample(0.1, ready_timeout=0.3)
    assert made[0].killed and not (tmp_path / "_voice_sample").exists()


def test_record_voice_sample_kills_a_hung_capture_and_waits_before_cleanup(tmp_path, monkeypatch):
    c, made = _voice_ctrl(tmp_path, monkeypatch, pcm=np.ones(100, np.int16).tobytes(), hang=True)
    audio = c.record_voice_sample(0.05, stop_timeout=0.2)
    assert made[0].events[-2:] == ["kill", "wait"] and len(audio) == 100     # killed, then waited for it to die, then read/deleted
    assert not (tmp_path / "_voice_sample").exists()


def test_delete_meeting_waits_for_a_running_voice_id(tmp_path, monkeypatch):
    d = _meeting(tmp_path, monkeypatch, "done")
    (d / "raw" / "mic.pcm").write_bytes(np.ones(1000, np.int16).tobytes())
    held = [np.memmap(d / "raw" / "mic.pcm", dtype=np.int16, mode="r")]       # what identify_speakers holds open
    c = Controller(Settings()); c._voiceid_running.add(d)

    def finish():
        time.sleep(1.5); held.clear(); c._voiceid_running.discard(d)
    threading.Thread(target=finish, daemon=True).start()
    assert c.delete_meeting(d) is True and not d.exists()


def test_recover_does_not_respawn_a_worker_without_end_marker(tmp_path, monkeypatch):
    """Without raw/END a worker would poll for more audio forever."""
    d = _meeting(tmp_path, monkeypatch, "listening", ended=False)
    store.update_meta(d, ended_at=time.time())
    calls = _recording_spawn(monkeypatch)
    Controller(Settings()).recover()
    assert calls == []
