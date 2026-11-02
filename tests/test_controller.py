import json, os, subprocess, sys, time

from meetassist import controller, paths, store
from meetassist.controller import Controller
from meetassist.settings import Settings


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


def test_recover_does_not_respawn_a_worker_without_end_marker(tmp_path, monkeypatch):
    """Without raw/END a worker would poll for more audio forever."""
    d = _meeting(tmp_path, monkeypatch, "listening", ended=False)
    store.update_meta(d, ended_at=time.time())
    calls = _recording_spawn(monkeypatch)
    Controller(Settings()).recover()
    assert calls == []
