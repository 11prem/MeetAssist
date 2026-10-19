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
        {"seq": 2, "track": "mic", "t_start": 1.5, "t_end": 2.5},     # 0.5s gap -> same turn
        {"seq": 3, "track": "mic", "t_start": 10.0, "t_end": 11.0},   # big gap -> new turn
    ]
    turns = voiceid._merge_turns(mic_rows)
    assert len(turns) == 2
    assert turns[0]["seqs"] == [1, 2] and turns[0]["t_end"] == 2.5
    assert turns[1]["seqs"] == [3]


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
