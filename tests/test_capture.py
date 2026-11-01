import numpy as np

from meetassist.capture_proc import RATE, _Track

EPOCH = 1000.0


def _chunks(n_chunks=5, secs=0.1):
    return [np.full(int(RATE * secs), 7, np.int16) for _ in range(n_chunks)]


def _track(tmp_path):
    return _Track("mic", tmp_path / "mic.pcm", EPOCH)


def _read(tmp_path):
    return np.frombuffer((tmp_path / "mic.pcm").read_bytes(), np.int16)


def test_late_but_complete_mic_audio_is_not_padded(tmp_path):
    """0.5 s of audio delivered 0.25 s late (jitter under PAD_THRESHOLD_S) must land contiguously, with no zeros inserted before/between chunks."""
    tr = _track(tmp_path)
    for c in _chunks():
        tr.q.put(c)
    tr.drain(EPOCH + 0.5 + 0.25)
    tr.f.close()
    x = _read(tmp_path)
    assert len(x) == int(0.5 * RATE) and (x == 7).all()


def test_real_gap_is_still_padded(tmp_path):
    """Samples that are really missing (backlog drained 2 s after the audio should have started) keep the timeline aligned."""
    tr = _track(tmp_path)
    for c in _chunks():
        tr.q.put(c)
    tr.drain(EPOCH + 2.5)
    tr.f.close()
    x = _read(tmp_path)
    assert len(x) == int(2.5 * RATE) and (x[: int(2.0 * RATE)] == 0).all() and (x[int(2.0 * RATE):] == 7).all()


def test_drain_on_empty_queue_is_a_noop(tmp_path):
    tr = _track(tmp_path)
    tr.drain(EPOCH + 5)
    assert tr.written == 0
