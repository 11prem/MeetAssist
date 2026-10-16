import numpy as np, av, pytest
from meetassist.segmenter import TrackSegmenter, RATE

def load(path):
    c = av.open(path); r = av.AudioResampler(format="s16", layout="mono", rate=RATE); out=[]
    for f in c.decode(audio=0):
        for x in r.resample(f): out.append(x.to_ndarray().reshape(-1))
    return np.concatenate(out).astype(np.float32)/32768

SAMPLE = r"C:\Users\prem.b\work\MeetAssist\audio\Recording.m4a"

def run(audio, block):
    seg = TrackSegmenter("mic"); us = []
    for k in range(0, len(audio), block): us += seg.feed(audio[k:k+block])
    us += seg.flush(); return us

def test_silence_gives_nothing():
    assert run(np.zeros(RATE*10, np.float32), 1600) == []

def test_real_speech_is_cut_at_pauses_and_bounded():
    a = load(SAMPLE)
    us = run(a, 1600)
    assert len(us) >= 5
    assert all(u.dur <= 29.0 for u in us)
    assert all(b.start_s >= a_.end_s - 0.6 for a_, b in zip(us, us[1:]))      # ordered, ~non-overlapping
    assert all(0 <= u.start_s < u.end_s <= len(a)/RATE + 0.1 for u in us)

def test_block_size_independent():
    a = load(SAMPLE)[:RATE*60]
    x, y = run(a, 1600), run(a, 8000)
    assert abs(len(x) - len(y)) <= 1

def test_forced_cuts_do_not_overlap():
    a = load(SAMPLE)
    seg = TrackSegmenter("mic", max_s=6.0, min_silence_s=0.5); us = []
    for k in range(0, len(a), 1600): us += seg.feed(a[k:k+1600])
    us += seg.flush()
    assert max(u.dur for u in us) <= 6.5
    assert all(b.start_s >= x.end_s - 1e-6 for x, b in zip(us, us[1:])), "adjacent utterances must not overlap (duplicate words)"
