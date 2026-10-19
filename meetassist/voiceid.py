"""Speaker voice identification: embeds mic-track utterances and matches/clusters them against
saved voice profiles (%LOCALAPPDATA%\\MeetAssist\\voices.json) so the same person is recognised
across meetings. Runs the CAM++ speaker-embedding model (models/speaker-id/*.onnx) on onnxruntime
and kaldi-native-fbank for features - no torch, no network (AGENTS rule 3). See models/speaker-id/README.txt
for where the model came from.
"""
from __future__ import annotations

import json
import os
import uuid
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from . import paths
from .segmenter import RATE

MATCH_THRESHOLD = 0.35          # shortcut: cosine-distance cutoff guessed, not calibrated on real
                                 # meetings; if speakers get merged/split wrongly, tune this from data.
MIN_SAMPLES = RATE // 4          # need >= 0.25s of audio for a stable embedding
ME_ID = "me"

_embedder: Optional["SpeakerEmbedder"] = None


def _fbank(audio: np.ndarray) -> np.ndarray:
    """float32 mono 16 kHz -> (T, 80) Kaldi-style fbank, mean-normalised (matches the model's
    own 'global-mean' training convention, see its ONNX metadata)."""
    import kaldi_native_fbank as knf
    opts = knf.FbankOptions()
    opts.frame_opts.samp_freq = RATE
    opts.frame_opts.dither = 0.0
    opts.frame_opts.window_type = "povey"
    opts.mel_opts.num_bins = 80
    fb = knf.OnlineFbank(opts)
    fb.accept_waveform(RATE, audio.astype(np.float32).tolist())
    fb.input_finished()
    n = fb.num_frames_ready
    if n == 0:
        return np.zeros((0, 80), np.float32)
    feats = np.stack([fb.get_frame(i) for i in range(n)]).astype(np.float32)
    return feats - feats.mean(axis=0, keepdims=True)


class SpeakerEmbedder:
    def __init__(self, model_path: Path) -> None:
        import onnxruntime as ort
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 1
        self._sess = ort.InferenceSession(str(model_path), sess_options=opts, providers=["CPUExecutionProvider"])
        self._input = self._sess.get_inputs()[0].name

    def embed(self, audio: np.ndarray) -> Optional[np.ndarray]:
        """audio: float32 mono 16 kHz, any amplitude scale. Returns an L2-normalised 192-d
        embedding, or None if there isn't enough audio for a stable one."""
        if len(audio) < MIN_SAMPLES:
            return None
        feats = _fbank(audio)
        if len(feats) == 0:
            return None
        (emb,) = self._sess.run(None, {self._input: feats[None]})
        v = emb[0]
        n = np.linalg.norm(v)
        return v / n if n > 1e-9 else None


def get_embedder() -> Optional[SpeakerEmbedder]:
    """Lazily loads the model; returns None if it isn't installed (feature silently unavailable,
    same convention as the optional Tamil model)."""
    global _embedder
    if _embedder is None and paths.SPEAKER_MODEL.exists():
        _embedder = SpeakerEmbedder(paths.SPEAKER_MODEL)
    return _embedder


def cosine_dist(a: np.ndarray, b: np.ndarray) -> float:
    return 1.0 - float(np.dot(a, b))


# ---------------------------------------------------------------- global profile store
def load_profiles() -> List[dict]:
    try:
        data = json.loads(paths.VOICES_FILE.read_text(encoding="utf-8"))
    except Exception:
        data = None
    return (data or {}).get("profiles", [])


def save_profiles(profiles: List[dict]) -> None:
    paths.VOICES_FILE.parent.mkdir(parents=True, exist_ok=True)
    tmp = paths.VOICES_FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps({"profiles": profiles}, indent=2), encoding="utf-8")
    os.replace(tmp, paths.VOICES_FILE)


def profile_names() -> Dict[str, str]:
    return {p["id"]: p["name"] for p in load_profiles() if p.get("name")}


def rename_profile(profile_id: str, name: str) -> None:
    profiles = load_profiles()
    for p in profiles:
        if p["id"] == profile_id:
            p["name"] = name
            break
    save_profiles(profiles)


def delete_profile(profile_id: str) -> None:
    save_profiles([p for p in load_profiles() if p["id"] != profile_id])


def enroll_me(audio: np.ndarray) -> bool:
    """One-time self-enrollment: (re)creates the 'me' profile's voiceprint from a fresh sample,
    so a future mic-track voice doesn't have to be assumed to be the user's - it's recognised."""
    embedder = get_embedder()
    emb = embedder.embed(audio) if embedder else None
    if emb is None:
        return False
    profiles = [p for p in load_profiles() if p["id"] != ME_ID]
    profiles.append({"id": ME_ID, "name": "Me", "centroid": emb.tolist(), "count": 1})
    save_profiles(profiles)
    return True


# ---------------------------------------------------------------- clustering
def _merge_centroid(c: dict, emb: np.ndarray) -> None:
    n = c["count"]
    centroid = (np.asarray(c["centroid"], dtype=np.float32) * n + emb) / (n + 1)
    norm = np.linalg.norm(centroid)
    c["centroid"] = (centroid / norm).tolist() if norm > 1e-9 else centroid.tolist()
    c["count"] = n + 1


def _cluster_indices(embeddings: List[np.ndarray], threshold: float) -> List[List[int]]:
    """Groups embedding indices by repeatedly merging the two closest groups (centroid linkage)
    while they're within `threshold` cosine distance. Order-independent, unlike a single greedy
    pass - needed so e.g. three short turns from the same unrecognised person become one group,
    not three (fine at one-meeting scale: tens of turns, not thousands)."""
    groups = [[i] for i in range(len(embeddings))]
    centroids = [embeddings[i] for i in range(len(embeddings))]
    while len(groups) > 1:
        bi, bj, bd = -1, -1, 1e9
        for i in range(len(groups)):
            for j in range(i + 1, len(groups)):
                d = cosine_dist(centroids[i], centroids[j])
                if d < bd:
                    bi, bj, bd = i, j, d
        if bd > threshold:
            break
        groups[bi] += groups[bj]
        merged = np.mean([embeddings[k] for k in groups[bi]], axis=0)
        centroids[bi] = merged / (np.linalg.norm(merged) + 1e-9)
        del groups[bj]; del centroids[bj]
    return groups


def assign_speakers(embeddings: List[np.ndarray], profiles: List[dict],
                     threshold: float = MATCH_THRESHOLD) -> Tuple[List[str], List[dict]]:
    """Matches each embedding to the nearest saved profile within `threshold` cosine distance,
    refining its centroid by a running mean. Embeddings that match no profile are then clustered
    among themselves (see _cluster_indices) so unrecognised turns from the same new person become
    one new, unnamed profile instead of several. Returns the profile id assigned to each embedding
    and the updated profile list."""
    updated = [dict(p) for p in profiles]
    assigned: List[Optional[str]] = [None] * len(embeddings)
    unmatched: List[int] = []
    for idx, emb in enumerate(embeddings):
        best_i, best_d = -1, 1e9
        for i, c in enumerate(updated):
            d = cosine_dist(emb, np.asarray(c["centroid"], dtype=np.float32))
            if d < best_d:
                best_d, best_i = d, i
        if best_i >= 0 and best_d <= threshold:
            _merge_centroid(updated[best_i], emb)
            assigned[idx] = updated[best_i]["id"]
        else:
            unmatched.append(idx)
    for group in _cluster_indices([embeddings[i] for i in unmatched], threshold):
        members = [unmatched[g] for g in group]
        centroid = np.mean([embeddings[i] for i in members], axis=0)
        centroid = centroid / (np.linalg.norm(centroid) + 1e-9)
        new_id = uuid.uuid4().hex[:12]
        updated.append({"id": new_id, "name": None, "centroid": centroid.tolist(), "count": len(members)})
        for i in members:
            assigned[i] = new_id
    return assigned, updated


def _merge_turns(mic_rows: List[dict], max_gap: float = 2.0) -> List[dict]:
    """Consecutive mic utterances close in time are almost always the same person talking
    continuously; merging them before embedding gives far fewer, much more stable embeddings than
    embedding every short VAD-cut utterance on its own (which was fragmenting one real speaker
    into several near-miss clusters and spamming the naming prompt)."""
    turns: List[dict] = []
    for r in mic_rows:
        t = turns[-1] if turns else None
        if t and r["t_start"] - t["t_end"] <= max_gap:
            t["t_end"] = max(t["t_end"], r["t_end"])
            t["seqs"].append(r["seq"])
        else:
            turns.append({"t_start": r["t_start"], "t_end": r["t_end"], "seqs": [r["seq"]]})
    return turns


def identify_speakers(meeting_dir: Path) -> List[str]:
    """Post-meeting pass: embeds each mic-track speaking turn, matches/clusters it against saved
    voice profiles, writes <meeting>/speakers.json (seq -> profile id) and updates the global
    profile store. Returns the ids of speakers in this meeting that still have no name, for the
    UI to prompt for (once per speaker, not once per utterance). No-ops quietly if the speaker-id
    model isn't installed."""
    from . import export, store  # local import: store.read_rows() calls back into this module

    embedder = get_embedder()
    if embedder is None:
        return []
    rows = export.read_journal(meeting_dir / "journal.jsonl")
    mic_rows = [r for r in rows if r["track"] == "mic"]
    raw = meeting_dir / "raw" / "mic.pcm"
    if not mic_rows or not raw.exists():
        return []
    pcm = np.memmap(raw, dtype=np.int16, mode="r")
    turns = _merge_turns(mic_rows)
    embeddings, turns_with_emb = [], []
    for t in turns:
        a, b = int(t["t_start"] * RATE), min(int(t["t_end"] * RATE), len(pcm))
        if b <= a:
            continue
        emb = embedder.embed(pcm[a:b].astype(np.float32) / 32768.0)
        if emb is not None:
            embeddings.append(emb)
            turns_with_emb.append(t)
    if not embeddings:
        return []
    profiles = load_profiles()
    assigned, updated = assign_speakers(embeddings, profiles)
    save_profiles(updated)
    speakers: Dict[int, str] = {}
    for turn, pid in zip(turns_with_emb, assigned):
        for seq in turn["seqs"]:
            speakers[seq] = pid
    store.write_speakers(meeting_dir, speakers)
    named_ids = {p["id"] for p in updated if p.get("name")}
    return sorted({pid for pid in assigned if pid not in named_ids})
