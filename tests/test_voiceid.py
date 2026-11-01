import json
import numpy as np
import pytest
from meetassist import export, paths, store, voiceid


def _unit(v):
    v = np.asarray(v, dtype=np.float32)
    return v / np.linalg.norm(v)


A = _unit([1.0, 0.0, 0.0])
A2 = _unit([0.98, 0.1, 0.0])     # close to A -> same speaker
B = _unit([0.0, 1.0, 0.0])       # orthogonal -> a different speaker


def test_assign_speakers_groups_similar_and_splits_dissimilar():
    ids, profiles = voiceid.assign_speakers([A, A2, B], [])
    assert ids[0] == ids[1] != ids[2]
    assert len(profiles) == 2
    assert all(p["name"] is None for p in profiles)


def test_assign_speakers_matches_and_refines_existing_profile():
    existing = [{"id": "p1", "name": "Alice", "centroid": A.tolist(), "count": 1}]
    ids, profiles = voiceid.assign_speakers([A2, B], existing)
    assert ids[0] == "p1" and ids[1] != "p1"
    alice = next(p for p in profiles if p["id"] == "p1")
    assert alice["count"] == 2
    assert abs(np.linalg.norm(np.asarray(alice["centroid"])) - 1.0) < 1e-5  # stays unit-norm


def test_assign_speakers_does_not_fragment_one_new_speaker_into_many(monkeypatch):
    """Regression for the reported bug: many short turns from the SAME unrecognised person used to
    spawn a separate unnamed profile each (and a naming prompt each), because the old greedy pass
    compared turns only to clusters seen so far, in order. Slight per-turn variation (just like real
    short utterances) must still collapse into one profile."""
    rng = np.random.default_rng(0)
    same_speaker = [_unit(A + rng.normal(0, 0.05, 3)) for _ in range(6)]
    ids, profiles = voiceid.assign_speakers(same_speaker, [])
    assert len(set(ids)) == 1
    assert len(profiles) == 1


def test_assign_speakers_keeps_three_interleaved_speakers_separate():
    rng = np.random.default_rng(1)
    C = _unit([0.0, 0.0, 1.0])
    noisy = lambda base: _unit(base + rng.normal(0, 0.03, 3))
    turns = [noisy(A), noisy(B), noisy(C), noisy(A), noisy(B), noisy(C), noisy(A)]
    ids, profiles = voiceid.assign_speakers(turns, [])
    assert len(set(ids)) == 3
    assert len(profiles) == 3
    assert ids[0] == ids[3] == ids[6]        # the three A turns all landed on the same profile


def test_merge_turns_joins_close_utterances_and_keeps_far_ones_apart():
    mic_rows = [
        {"seq": 1, "track": "mic", "t_start": 0.0, "t_end": 1.0},
        {"seq": 2, "track": "mic", "t_start": 1.4, "t_end": 2.4},     # 0.4s gap -> same turn
        {"seq": 3, "track": "mic", "t_start": 3.0, "t_end": 4.0},     # 0.6s gap -> new turn (two people back-to-back)
        {"seq": 4, "track": "mic", "t_start": 10.0, "t_end": 11.0},   # big gap -> new turn
    ]
    turns = voiceid._merge_turns(mic_rows)
    assert len(turns) == 3
    assert turns[0]["seqs"] == [1, 2] and turns[0]["t_end"] == 2.4
    assert turns[1]["seqs"] == [3] and turns[2]["seqs"] == [4]


def test_merge_turns_caps_turn_length():
    rows = [{"seq": i, "track": "mic", "t_start": i * 5.0, "t_end": i * 5.0 + 4.8} for i in range(6)]  # gapless-ish, 30 s
    turns = voiceid._merge_turns(rows)
    assert [t["seqs"] for t in turns] == [[0, 1, 2], [3, 4, 5]]      # 14.8 s then a new turn, never past 15 s
    assert all(t["t_end"] - t["t_start"] <= 15.0 for t in turns)


def test_profile_store_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "VOICES_FILE", tmp_path / "voices.json")
    voiceid.save_profiles([{"id": "p1", "name": "Alice", "centroid": A.tolist(), "count": 1},
                           {"id": "p2", "name": None, "centroid": B.tolist(), "count": 1}])
    assert voiceid.profile_names() == {"p1": "Alice"}
    voiceid.rename_profile("p2", "Bob")
    assert voiceid.profile_names() == {"p1": "Alice", "p2": "Bob"}
    voiceid.delete_profile("p1")
    assert [p["id"] for p in voiceid.load_profiles()] == ["p2"]


def test_attach_speakers_and_grouping_use_recognised_names():
    rows = [
        {"seq": 1, "track": "mic", "t_start": 1.0, "t_end": 2.0, "text": "hello"},
        {"seq": 2, "track": "mic", "t_start": 2.5, "t_end": 3.0, "text": "there"},
        {"seq": 3, "track": "mic", "t_start": 4.0, "t_end": 5.0, "text": "hi"},
    ]
    export.attach_speakers(rows, {1: "p1", 2: "p1", 3: "p2"}, {"p1": "Alice", "p2": "Bob"})
    groups = export.group_rows(rows)
    assert [g["who"] for g in groups] == ["Alice", "Bob"]
    assert groups[0]["text"] == "hello there"
    txt = export.render_txt(rows)
    assert "[Alice]" in txt and "[Bob]" in txt


def test_attach_speakers_falls_back_to_track_label_when_unnamed():
    rows = [{"seq": 1, "track": "mic", "t_start": 1.0, "t_end": 2.0, "text": "hello"}]
    export.attach_speakers(rows, {1: "p1"}, {})          # p1 exists but has no name yet
    assert "who" not in rows[0]
    assert "[You]" in export.render_txt(rows)


@pytest.mark.skipif(not paths.SPEAKER_MODEL.exists(), reason="speaker-id model not installed")
def test_embedder_produces_a_stable_unit_vector():
    embedder = voiceid.SpeakerEmbedder(paths.SPEAKER_MODEL)
    rng = np.random.default_rng(0)
    audio = rng.normal(0, 0.05, voiceid.MIN_SAMPLES * 4).astype(np.float32)
    a, b = embedder.embed(audio), embedder.embed(audio)
    assert a is not None and a.shape == (192,)
    assert abs(np.linalg.norm(a) - 1.0) < 1e-4
    assert np.allclose(a, b)                              # deterministic (no dither)
    assert embedder.embed(audio[: voiceid.MIN_SAMPLES // 2]) is None   # too short


@pytest.mark.skipif(not paths.SPEAKER_MODEL.exists(), reason="speaker-id model not installed")
def test_embedder_ignores_digital_silence():
    """Capture pads gaps/startup with exact zeros; that must not skew the voiceprint."""
    embedder = voiceid.SpeakerEmbedder(paths.SPEAKER_MODEL)
    rng = np.random.default_rng(2)
    t = np.arange(voiceid.RATE * 3) / voiceid.RATE
    env = 0.5 + 0.5 * np.sin(2 * np.pi * 3 * t)                      # syllable-like amplitude modulation
    voice = sum(np.sin(2 * np.pi * 140 * k * t) / k for k in range(1, 12)) * env * 0.1
    speech = (voice + rng.normal(0, 0.005, len(t))).astype(np.float32)
    pad = np.zeros(voiceid.RATE * 3, np.float32)
    plain, padded = embedder.embed(speech), embedder.embed(np.concatenate([pad, speech, pad]))
    assert plain is not None and padded is not None
    assert voiceid.cosine_dist(plain, padded) < 0.05
    assert embedder.embed(np.zeros(voiceid.RATE * 2, np.float32)) is None


@pytest.mark.skipif(not paths.SPEAKER_MODEL.exists(), reason="speaker-id model not installed")
def test_identify_speakers_writes_speakers_json(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "VOICES_FILE", tmp_path / "voices.json")
    d = tmp_path / "meeting"
    (d / "raw").mkdir(parents=True)
    rng = np.random.default_rng(1)
    sr = 16000
    audio = (rng.normal(0, 0.05, sr * 6)).astype(np.float32)
    (d / "raw" / "mic.pcm").write_bytes((audio * 32768).astype(np.int16).tobytes())
    rows = [
        {"seq": 1, "track": "mic", "t_start": 0.0, "t_end": 2.0, "text": "hello"},
        {"seq": 2, "track": "mic", "t_start": 2.0, "t_end": 4.0, "text": "there"},
    ]
    (d / "journal.jsonl").write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    unnamed = voiceid.identify_speakers(d)
    speakers = store.read_speakers(d)
    assert set(speakers.keys()) == {1, 2}
    assert unnamed == sorted(set(speakers.values()))       # nothing named yet


class _SignEmbedder:
    """Fake embedder: positive audio -> vector A, negative -> B (so each track gets its own 'voice')."""
    def embed(self, audio):
        return A if audio.mean() > 0 else B


def _two_track_meeting(tmp_path, with_loop=True):
    d = tmp_path / "meeting"
    (d / "raw").mkdir(parents=True)
    sr = voiceid.RATE
    (d / "raw" / "mic.pcm").write_bytes(np.full(sr * 6, 1000, np.int16).tobytes())
    if with_loop:
        (d / "raw" / "loop.pcm").write_bytes(np.full(sr * 6, -1000, np.int16).tobytes())
    rows = [
        {"seq": 1, "track": "mic", "t_start": 0.0, "t_end": 1.0, "text": "me"},
        {"seq": 2, "track": "loop", "t_start": 1.1, "t_end": 2.0, "text": "remote"},     # close to seq 1, still its own turn
        {"seq": 3, "track": "loop", "t_start": 2.2, "t_end": 3.0, "text": "remote again"},
    ]
    (d / "journal.jsonl").write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    return d


def test_identify_speakers_covers_both_tracks_without_merging_across_them(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "VOICES_FILE", tmp_path / "voices.json")
    monkeypatch.setattr(voiceid, "get_embedder", lambda: _SignEmbedder())
    d = _two_track_meeting(tmp_path)
    voiceid.identify_speakers(d)
    sp = store.read_speakers(d)
    assert set(sp) == {1, 2, 3}
    assert sp[2] == sp[3] != sp[1]
    assert len(voiceid.load_profiles()) == 2                  # ONE pooled assignment


def test_identify_speakers_skips_a_track_whose_pcm_is_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "VOICES_FILE", tmp_path / "voices.json")
    monkeypatch.setattr(voiceid, "get_embedder", lambda: _SignEmbedder())
    d = _two_track_meeting(tmp_path, with_loop=False)
    voiceid.identify_speakers(d)
    assert set(store.read_speakers(d)) == {1}


def test_find_samples_records_track_and_sample_audio_reads_loop_pcm(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "MEETINGS_DIR", tmp_path / "m")
    d = store.new_meeting("Remote")
    store.update_meta(d, epoch=1000)
    (d / "journal.jsonl").write_text(json.dumps({"seq": 1, "track": "loop", "lang": "en", "t_start": 1, "t_end": 3, "text": "hi"}),
                                     encoding="utf-8")
    store.write_speakers(d, {1: "p1"})
    (d / "raw" / "mic.pcm").write_bytes(np.full(voiceid.RATE * 4, 1000, np.int16).tobytes())
    (d / "raw" / "loop.pcm").write_bytes(np.full(voiceid.RATE * 4, -2000, np.int16).tobytes())
    s = voiceid.find_samples()["p1"]
    assert s["track"] == "loop"
    clip = voiceid.sample_audio(s)
    assert len(clip) == 2 * voiceid.RATE and np.allclose(clip, -2000 / 32768.0)
    assert np.allclose(voiceid.sample_audio({k: v for k, v in s.items() if k != "track"}), 1000 / 32768.0)   # default mic


def _meeting_with_voices(title, epoch, rows, speakers, audio_s=None):
    d = store.new_meeting(title)
    store.update_meta(d, epoch=epoch)
    (d / "journal.jsonl").write_text("\n".join(json.dumps({"track": "mic", "lang": "en", **r}) for r in rows), encoding="utf-8")
    store.write_speakers(d, speakers)
    if audio_s:
        (d / "raw" / "mic.pcm").write_bytes((np.arange(voiceid.RATE * audio_s) % 100).astype(np.int16).tobytes())
    return d


def test_find_samples_uses_newest_meeting_and_longest_turn(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "MEETINGS_DIR", tmp_path / "m")
    _meeting_with_voices("Old sync", 1000, [{"seq": 1, "t_start": 0, "t_end": 9, "text": "from the old one"}], {1: "p1"})
    new = _meeting_with_voices("New sync", 2000, [
        {"seq": 1, "t_start": 0, "t_end": 1, "text": "short"},
        {"seq": 2, "t_start": 5, "t_end": 9, "text": "long one"},
        {"seq": 3, "t_start": 10, "t_end": 11, "text": "someone else"}], {1: "p1", 2: "p1", 3: "p2"}, audio_s=12)
    found = voiceid.find_samples()
    assert set(found) == {"p1", "p2"}
    assert found["p1"]["title"] == "New sync" and found["p1"]["meeting"] == new
    assert (found["p1"]["t_start"], found["p1"]["t_end"]) == (5, 9)
    assert found["p1"]["text"] == "short long one"
    clip = voiceid.sample_audio(found["p1"])
    assert clip.dtype == np.float32 and len(clip) == 4 * voiceid.RATE
    assert np.isclose(clip[0], (5 * voiceid.RATE % 100) / 32768.0)            # the right slice of the raw audio


def test_sample_audio_is_none_when_raw_audio_is_gone(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "MEETINGS_DIR", tmp_path / "m")
    _meeting_with_voices("No audio", 1000, [{"seq": 1, "t_start": 0, "t_end": 2, "text": "hi"}], {1: "p1"})
    assert voiceid.sample_audio(voiceid.find_samples()["p1"]) is None


def test_rename_during_identify_is_not_lost(tmp_path, monkeypatch):
    """#7: a Settings rename landing between identify's load and save used to be overwritten by identify's save."""
    import threading
    monkeypatch.setattr(paths, "VOICES_FILE", tmp_path / "voices.json")
    monkeypatch.setattr(voiceid, "get_embedder", lambda: _SignEmbedder())
    voiceid.save_profiles([{"id": "p0", "name": None, "centroid": A.tolist(), "count": 1}])
    d = _two_track_meeting(tmp_path)
    real, renamer = voiceid.assign_speakers, []

    def racing_assign(embs, profiles, *a, **k):
        t = threading.Thread(target=voiceid.rename_profile, args=("p0", "Zed"))
        t.start(); renamer.append(t)
        t.join(0.3)                      # unlocked: finishes (and saves) here; locked: parked on the RLock
        return real(embs, profiles, *a, **k)
    monkeypatch.setattr(voiceid, "assign_speakers", racing_assign)
    voiceid.identify_speakers(d)
    renamer[0].join(5)
    assert not renamer[0].is_alive()
    assert voiceid.profile_names() == {"p0": "Zed"}


def _two_meetings(monkeypatch, tmp_path, maps):
    monkeypatch.setattr(paths, "MEETINGS_DIR", tmp_path / "m")
    monkeypatch.setattr(paths, "VOICES_FILE", tmp_path / "voices.json")
    ds = []
    for i, mp in enumerate(maps):
        d = store.new_meeting(f"m{i}"); store.write_speakers(d, mp); ds.append(d)
    return ds


def test_rename_to_an_existing_name_merges_profiles(tmp_path, monkeypatch):
    """D6: same name on two profiles -> one profile (survivor keeps its id), weighted unit centroid, speakers.json remapped."""
    d1, d2 = _two_meetings(monkeypatch, tmp_path, [{1: "p2", 2: "p3"}, {1: "p1", 2: "p2"}])
    voiceid.save_profiles([{"id": "p1", "name": "Alice", "centroid": A.tolist(), "count": 2},
                           {"id": "p2", "name": None, "centroid": B.tolist(), "count": 1},
                           {"id": "p3", "name": None, "centroid": B.tolist(), "count": 1}])
    assert voiceid.rename_profile("p2", "  alice ") == "p1"
    ps = {p["id"]: p for p in voiceid.load_profiles()}
    assert set(ps) == {"p1", "p3"} and ps["p1"]["name"] == "Alice" and ps["p1"]["count"] == 3
    want = _unit(2 * A + B)
    assert np.allclose(ps["p1"]["centroid"], want, atol=1e-5) and abs(np.linalg.norm(ps["p1"]["centroid"]) - 1) < 1e-5
    assert store.read_speakers(d1) == {1: "p1", 2: "p3"} and store.read_speakers(d2) == {1: "p1", 2: "p1"}


def test_rename_to_a_new_name_returns_same_id_without_merging(tmp_path, monkeypatch):
    _two_meetings(monkeypatch, tmp_path, [])
    voiceid.save_profiles([{"id": "p1", "name": "Alice", "centroid": A.tolist(), "count": 1},
                           {"id": "p2", "name": None, "centroid": B.tolist(), "count": 1}])
    assert voiceid.rename_profile("p2", "Bob") == "p2"
    assert voiceid.profile_names() == {"p1": "Alice", "p2": "Bob"}


def test_delete_profile_removes_it_from_every_speakers_json(tmp_path, monkeypatch):
    """D7: a deleted voice must not stay referenced by meetings."""
    d1, d2 = _two_meetings(monkeypatch, tmp_path, [{1: "p1", 2: "p2"}, {1: "p1"}])
    voiceid.save_profiles([{"id": "p1", "name": None, "centroid": A.tolist(), "count": 1},
                           {"id": "p2", "name": None, "centroid": B.tolist(), "count": 1}])
    voiceid.delete_profile("p1")
    assert store.read_speakers(d1) == {2: "p2"} and store.read_speakers(d2) == {}
    assert [p["id"] for p in voiceid.load_profiles()] == ["p2"]


def _failing_second_write(monkeypatch):
    real, n = store.write_speakers, []
    def flaky(d, mapping):
        n.append(d)
        if len(n) == 2:
            raise PermissionError("locked by AV")
        real(d, mapping)
    monkeypatch.setattr(store, "write_speakers", flaky)
    return lambda: monkeypatch.setattr(store, "write_speakers", real)


def test_merge_remap_failure_leaves_profiles_intact_and_retryable(tmp_path, monkeypatch):
    """A speakers.json write failing midway must not leave meetings pointing at a profile that was already removed."""
    d1, d2 = _two_meetings(monkeypatch, tmp_path, [{1: "p2"}, {1: "p2"}])
    profs = [{"id": "p1", "name": "Alice", "centroid": A.tolist(), "count": 1},
             {"id": "p2", "name": None, "centroid": B.tolist(), "count": 1}]
    voiceid.save_profiles(profs)
    heal = _failing_second_write(monkeypatch)
    with pytest.raises(PermissionError):
        voiceid.rename_profile("p2", "Alice")
    assert voiceid.load_profiles() == profs                    # p2 still there -> the user can retry
    heal()
    assert voiceid.rename_profile("p2", "Alice") == "p1"
    assert [p["id"] for p in voiceid.load_profiles()] == ["p1"] and voiceid.load_profiles()[0]["count"] == 2
    assert store.read_speakers(d1) == {1: "p1"} and store.read_speakers(d2) == {1: "p1"}


def test_delete_remap_failure_leaves_profiles_intact_and_retryable(tmp_path, monkeypatch):
    d1, d2 = _two_meetings(monkeypatch, tmp_path, [{1: "p1"}, {1: "p1"}])
    profs = [{"id": "p1", "name": None, "centroid": A.tolist(), "count": 1}]
    voiceid.save_profiles(profs)
    heal = _failing_second_write(monkeypatch)
    with pytest.raises(PermissionError):
        voiceid.delete_profile("p1")
    assert voiceid.load_profiles() == profs
    heal()
    voiceid.delete_profile("p1")
    assert voiceid.load_profiles() == [] and store.read_speakers(d1) == {} == store.read_speakers(d2)


def test_identify_speakers_saves_nothing_when_the_journal_is_replaced_mid_run(tmp_path, monkeypatch):
    """Refine swaps in a journal with different seqs while identify (started on the draft) runs: its result is stale."""
    monkeypatch.setattr(paths, "VOICES_FILE", tmp_path / "voices.json")
    monkeypatch.setattr(voiceid, "get_embedder", lambda: _SignEmbedder())
    initial = [{"id": "p0", "name": None, "centroid": A.tolist(), "count": 1}]
    voiceid.save_profiles(initial)
    d = _two_track_meeting(tmp_path)
    real, swap = voiceid.assign_speakers, [True]

    def swapping_assign(embs, profiles, *a, **k):
        if swap[0]:
            new = [{"seq": 10, "track": "mic", "t_start": 0.0, "t_end": 1.0, "text": "refined"},
                   {"seq": 11, "track": "mic", "t_start": 1.2, "t_end": 2.0, "text": "refined too"}]
            (d / "journal.jsonl").write_text("\n".join(json.dumps(r) for r in new), encoding="utf-8")
        return real(embs, profiles, *a, **k)
    monkeypatch.setattr(voiceid, "assign_speakers", swapping_assign)
    assert voiceid.identify_speakers(d) is None
    assert not (d / "speakers.json").exists() and voiceid.load_profiles() == initial
    swap[0] = False
    assert voiceid.identify_speakers(d) is not None
    assert set(store.read_speakers(d)) == {10, 11}


def _ref_cluster(embeddings, threshold):
    """The original pure-Python centroid-linkage clustering, kept as the reference semantics."""
    groups = [[i] for i in range(len(embeddings))]
    centroids = list(embeddings)
    while len(groups) > 1:
        bi, bj, bd = -1, -1, 1e9
        for i in range(len(groups)):
            for j in range(i + 1, len(groups)):
                d = voiceid.cosine_dist(centroids[i], centroids[j])
                if d < bd:
                    bi, bj, bd = i, j, d
        if bd > threshold:
            break
        groups[bi] += groups[bj]
        merged = np.mean([embeddings[k] for k in groups[bi]], axis=0)
        centroids[bi] = merged / (np.linalg.norm(merged) + 1e-9)
        del groups[bj]; del centroids[bj]
    return groups


def _speaker_cloud(rng, n, speakers, sigmas):
    centers = [_unit(rng.normal(size=192)) for _ in range(speakers)]
    return [_unit(centers[rng.integers(speakers)] + rng.choice(sigmas) * rng.normal(size=192) / np.sqrt(192))
            for _ in range(n)]


def test_cluster_indices_matches_the_reference_implementation():
    for seed in range(4):
        rng = np.random.default_rng(seed)
        embs = _speaker_cloud(rng, 60, 6, [0.5, 0.7, 0.75, 0.8, 1.0])       # mix of easy and near-threshold merges
        got, want = voiceid._cluster_indices(embs, voiceid.MATCH_THRESHOLD), _ref_cluster(embs, voiceid.MATCH_THRESHOLD)
        assert sorted(map(sorted, got)) == sorted(map(sorted, want))
        assert 1 < len(want) < len(embs)                                       # the data really exercises merging


def test_cluster_indices_is_fast_on_a_long_meeting():
    import time
    rng = np.random.default_rng(0)
    embs = _speaker_cloud(rng, 600, 20, [0.4, 0.6])                            # a long real meeting: ~580 merges
    rand = [_unit(v) for v in rng.normal(size=(1000, 192))]                    # 1000 turns, no merges
    t0 = time.perf_counter()
    groups = voiceid._cluster_indices(embs, voiceid.MATCH_THRESHOLD)
    assert len(voiceid._cluster_indices(rand, voiceid.MATCH_THRESHOLD)) == 1000
    assert time.perf_counter() - t0 < 2.0
    assert len(groups) == 20


def test_rename_merges_every_profile_sharing_the_name(tmp_path, monkeypatch):
    """D6 heal: ThiruKarthik x3 must collapse to one, not leave a duplicate behind."""
    C = _unit([0.0, 0.0, 1.0])
    d1, d2 = _two_meetings(monkeypatch, tmp_path, [{1: "p1", 2: "p2"}, {1: "p3", 2: "p4"}])
    voiceid.save_profiles([{"id": "p1", "name": "Thiru", "centroid": A.tolist(), "count": 3},
                           {"id": "p2", "name": "thiru ", "centroid": B.tolist(), "count": 2},
                           {"id": "p3", "name": None, "centroid": C.tolist(), "count": 1},
                           {"id": "p4", "name": "Bob", "centroid": C.tolist(), "count": 1}])
    assert voiceid.rename_profile("p3", "THIRU") == "p1"
    ps = {p["id"]: p for p in voiceid.load_profiles()}
    assert set(ps) == {"p1", "p4"} and ps["p1"]["count"] == 6
    assert np.allclose(ps["p1"]["centroid"], _unit(3 * A + 2 * B + C), atol=1e-5)
    assert store.read_speakers(d1) == {1: "p1", 2: "p1"} and store.read_speakers(d2) == {1: "p1", 2: "p4"}


def test_merge_duplicate_names_heals_existing_duplicates(tmp_path, monkeypatch):
    C = _unit([0.0, 0.0, 1.0])
    d1, d2 = _two_meetings(monkeypatch, tmp_path, [{1: "t1", 2: "t2", 3: "g2"}, {1: "t3", 2: "x"}])
    voiceid.save_profiles([{"id": "t1", "name": "ThiruKarthik", "centroid": A.tolist(), "count": 9},
                           {"id": "g1", "name": "Ganesh V", "centroid": B.tolist(), "count": 4},
                           {"id": "t2", "name": "ThiruKarthik", "centroid": B.tolist(), "count": 4},
                           {"id": "x", "name": None, "centroid": C.tolist(), "count": 4},
                           {"id": "g2", "name": "ganesh v", "centroid": B.tolist(), "count": 4},
                           {"id": "t3", "name": "ThiruKarthik ", "centroid": C.tolist(), "count": 4}])
    voiceid.merge_duplicate_names()
    ps = {p["id"]: p for p in voiceid.load_profiles()}
    assert set(ps) == {"t1", "g1", "x"} and ps["t1"]["count"] == 17 and ps["g1"]["count"] == 8
    assert np.allclose(ps["t1"]["centroid"], _unit(9 * A + 4 * B + 4 * C), atol=1e-5)
    assert np.allclose(ps["g1"]["centroid"], B, atol=1e-5)
    assert store.read_speakers(d1) == {1: "t1", 2: "t1", 3: "g1"} and store.read_speakers(d2) == {1: "t1", 2: "x"}


def test_merge_duplicate_names_is_a_noop_on_a_clean_store(tmp_path, monkeypatch):
    _two_meetings(monkeypatch, tmp_path, [{1: "p1"}])
    voiceid.save_profiles([{"id": "p1", "name": "Alice", "centroid": A.tolist(), "count": 1},
                           {"id": "p2", "name": None, "centroid": B.tolist(), "count": 1},
                           {"id": "p3", "name": None, "centroid": B.tolist(), "count": 1}])
    before = paths.VOICES_FILE.read_bytes()
    saves = []
    monkeypatch.setattr(voiceid, "save_profiles", lambda p: saves.append(p))
    voiceid.merge_duplicate_names()
    assert saves == [] and paths.VOICES_FILE.read_bytes() == before


def test_rename_to_a_blank_name_changes_nothing(tmp_path, monkeypatch):
    _two_meetings(monkeypatch, tmp_path, [])
    profs = [{"id": "p1", "name": None, "centroid": A.tolist(), "count": 1},
             {"id": "p2", "name": None, "centroid": B.tolist(), "count": 1}]
    voiceid.save_profiles(profs)
    assert voiceid.rename_profile("p2", "   ") == "p2"
    assert voiceid.load_profiles() == profs                    # not merged into the other unnamed voice


def test_reidentifying_a_meeting_does_not_add_its_audio_to_profiles_twice(tmp_path, monkeypatch):
    """After refine/retranscribe the same audio is identified again: ids get assigned, centroids stay put."""
    monkeypatch.setattr(paths, "VOICES_FILE", tmp_path / "voices.json")
    monkeypatch.setattr(voiceid, "get_embedder", lambda: _SignEmbedder())
    d = _two_track_meeting(tmp_path)
    voiceid.identify_speakers(d)
    first = voiceid.load_profiles()
    assert store.read_meta(d).get("voiceid_done") is True
    (d / "speakers.json").unlink()                             # what refine/retranscribe do
    voiceid.identify_speakers(d)
    assert voiceid.load_profiles() == first
    assert set(store.read_speakers(d)) == {1, 2, 3}


def test_identify_speakers_tolerates_an_odd_byte_pcm(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "VOICES_FILE", tmp_path / "voices.json")
    monkeypatch.setattr(voiceid, "get_embedder", lambda: _SignEmbedder())
    d = _two_track_meeting(tmp_path, with_loop=False)
    with open(d / "raw" / "mic.pcm", "ab") as f:
        f.write(b"\x01")                                       # torn last sample (hard kill mid-write)
    voiceid.identify_speakers(d)
    assert set(store.read_speakers(d)) == {1}
