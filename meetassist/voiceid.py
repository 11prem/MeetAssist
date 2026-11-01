"""Speaker voice identification: embeds mic- and loopback-track utterances and matches/clusters them against
saved voice profiles (%LOCALAPPDATA%\\MeetAssist\\voices.json) so the same person is recognised
across meetings. Runs the CAM++ speaker-embedding model (models/speaker-id/*.onnx) on onnxruntime
and kaldi-native-fbank for features - no torch, no network (AGENTS rule 3). See models/speaker-id/README.txt
for where the model came from.
"""
from __future__ import annotations

import json
import os
import threading
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
_LOCK = threading.RLock()        # guards every load->modify->save of voices.json (voice-id thread vs Settings dialog)


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
        # Capture pads gaps/startup with exact zeros; their fbank is a huge constant that skews the
        # mean-normalisation, so drop every all-zero 10 ms block first.
        blocks = audio[: len(audio) // 160 * 160].reshape(-1, 160)
        audio = blocks[blocks.any(axis=1)].ravel()
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


def _rewrite_speakers(remap: Dict[str, Optional[str]]) -> None:
    """Points every meeting's speakers.json entries for each profile in `remap` at its new id (or drops them if None)."""
    from . import store  # local import: store.read_rows() calls back into this module
    for m in store.list_meetings():
        sp = store.read_speakers(m.path)
        if any(v in remap for v in sp.values()):
            store.write_speakers(m.path, {k: remap.get(v, v) for k, v in sp.items() if remap.get(v, v)})


def _name_key(p: dict) -> str:
    return (p.get("name") or "").strip().casefold()


def _merge_groups(profiles: List[dict], groups: List[List[dict]]) -> None:
    """Folds each group into its first profile (count-weighted unit centroid, counts summed), re-points meetings at
    the survivors, then saves. Remap first (idempotent): a failure leaves profiles intact, so a retry works."""
    remap: Dict[str, Optional[str]] = {}
    for keep, *rest in groups:
        c = sum(np.asarray(p["centroid"], dtype=np.float32) * p["count"] for p in (keep, *rest))
        norm = np.linalg.norm(c)
        keep["centroid"] = (c / norm if norm > 1e-9 else c).tolist()
        keep["count"] = sum(p["count"] for p in (keep, *rest))
        remap.update({p["id"]: keep["id"] for p in rest})
    _rewrite_speakers(remap)
    save_profiles([p for p in profiles if p["id"] not in remap])


def rename_profile(profile_id: str, name: str) -> str:
    """Names a voice. Every other profile with that name (case-insensitive) is the same person: all merge into
    one existing named profile (its id survives, meetings re-pointed). Returns the surviving id."""
    name = name.strip()
    if not name:                                       # "" would match (and swallow) every unnamed voice
        return profile_id
    with _LOCK:
        profiles = load_profiles()
        me = next((p for p in profiles if p["id"] == profile_id), None)
        if me is None:
            return profile_id
        same = sorted((p for p in profiles if p["id"] != profile_id and _name_key(p) == name.casefold()),
                      key=lambda p: p["id"] != ME_ID)  # the enrolled 'me' profile must never be merged away
        if not same:
            me["name"] = name
            save_profiles(profiles)
            return profile_id
        _merge_groups(profiles, [same + [me]])
        return same[0]["id"]


def merge_duplicate_names() -> None:
    """One-shot heal for stores saved before rename merged all same-name voices: folds each group of
    same-named profiles into one. Idempotent; saves nothing when there are no duplicates."""
    with _LOCK:
        profiles = load_profiles()
        by_name: Dict[str, List[dict]] = {}
        for p in profiles:
            if _name_key(p):
                by_name.setdefault(_name_key(p), []).append(p)
        dups = [sorted(g, key=lambda p: p["id"] != ME_ID) for g in by_name.values() if len(g) > 1]
        if dups:
            _merge_groups(profiles, dups)


def delete_profile(profile_id: str) -> None:
    with _LOCK:
        _rewrite_speakers({profile_id: None})          # before the save, same reason as _merge_groups
        save_profiles([p for p in load_profiles() if p["id"] != profile_id])


def find_samples() -> Dict[str, dict]:
    """Where each saved voice was heard, for Settings > Voices: the newest meeting it appears in and its
    longest utterance there (the clearest clip to listen to). Voices heard in no saved meeting are left out."""
    from . import export, store  # local import: store.read_rows() calls back into this module

    found: Dict[str, dict] = {}
    for m in store.list_meetings():
        speakers = store.read_speakers(m.path)
        if not speakers:
            continue
        by_voice: Dict[str, List[dict]] = {}
        for r in export.read_journal(m.path / "journal.jsonl"):
            pid = speakers.get(r.get("seq"))
            if pid and pid not in found:
                by_voice.setdefault(pid, []).append(r)
        for pid, turns in by_voice.items():
            longest = max(turns, key=lambda r: r["t_end"] - r["t_start"])
            found[pid] = {"meeting": m.path, "title": m.title, "t_start": longest["t_start"], "t_end": longest["t_end"],
                          "track": longest.get("track", "mic"), "text": " ".join(t.get("text", "") for t in turns[:3]).strip()[:220]}
    return found


def sample_audio(sample: dict) -> Optional[np.ndarray]:
    """The clip of a find_samples() entry as float32 mono 16 kHz, or None if that meeting's raw audio is gone."""
    pcm = Path(sample["meeting"]) / "raw" / f"{sample.get('track', 'mic')}.pcm"
    if not pcm.exists():
        return None
    a, b = int(sample["t_start"] * RATE), min(int(sample["t_end"] * RATE), pcm.stat().st_size // 2)
    if b <= a:
        return None
    return np.fromfile(pcm, dtype=np.int16, count=b - a, offset=2 * a).astype(np.float32) / 32768.0


def enroll_me(audio: np.ndarray) -> bool:
    """One-time self-enrollment: (re)creates the 'me' profile's voiceprint from a fresh sample,
    so a future mic-track voice doesn't have to be assumed to be the user's - it's recognised."""
    embedder = get_embedder()
    emb = embedder.embed(audio) if embedder else None
    if emb is None:
        return False
    with _LOCK:
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
    not three. Vectorised over a distance matrix: a meeting has hundreds of turns (600 ~0.2 s; 1000 all
    merging ~1.2 s)."""
    n = len(embeddings)
    if n < 2:
        return [[i] for i in range(n)]
    groups: List[Optional[List[int]]] = [[i] for i in range(n)]
    sums = np.asarray(embeddings, dtype=np.float64)     # per-group sum of member embeddings
    cent = sums.copy()
    alive = np.ones(n, dtype=bool)
    dist = 1.0 - cent @ cent.T
    np.fill_diagonal(dist, np.inf)
    while True:
        i, j = divmod(int(np.argmin(dist)), n)          # row-major: the first (lowest i, j) pair on a tie, as before
        if not dist[i, j] <= threshold:                 # also stops once everything is masked (inf)
            break
        i, j = min(i, j), max(i, j)
        groups[i] += groups[j]; groups[j] = None; alive[j] = False
        sums[i] += sums[j]
        c = sums[i] / len(groups[i])
        cent[i] = c / (np.linalg.norm(c) + 1e-9)
        dist[j, :] = dist[:, j] = np.inf
        row = 1.0 - cent @ cent[i]
        row[~alive] = np.inf; row[i] = np.inf
        dist[i, :] = dist[:, i] = row
    return [g for g in groups if g is not None]


def assign_speakers(embeddings: List[np.ndarray], profiles: List[dict],
                     threshold: float = MATCH_THRESHOLD, learn: bool = True) -> Tuple[List[str], List[dict]]:
    """Matches each embedding to the nearest saved profile within `threshold` cosine distance,
    refining its centroid by a running mean (unless `learn` is False). Embeddings that match no profile are
    then clustered among themselves (see _cluster_indices) so unrecognised turns from the same new person become
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
            if learn:
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


def _merge_turns(rows: List[dict], max_gap: float = 0.5, max_len: float = 15.0) -> List[dict]:
    """Consecutive same-track utterances close in time are almost always the same person talking
    continuously; merging them before embedding gives far fewer, much more stable embeddings than
    embedding every short VAD-cut utterance on its own (which was fragmenting one real speaker
    into several near-miss clusters and spamming the naming prompt). The gap is tight and turns
    are length-capped so two people talking back-to-back don't blend into one embedding."""
    turns: List[dict] = []
    for r in rows:
        t = turns[-1] if turns else None
        if t and r["t_start"] - t["t_end"] <= max_gap and max(t["t_end"], r["t_end"]) - t["t_start"] <= max_len:
            t["t_end"] = max(t["t_end"], r["t_end"])
            t["seqs"].append(r["seq"])
        else:
            turns.append({"t_start": r["t_start"], "t_end": r["t_end"], "seqs": [r["seq"]]})
    return turns


def _track_embeddings(embedder: SpeakerEmbedder, rows: List[dict], raw: Path) -> Tuple[List[np.ndarray], List[dict]]:
    """Embeds each speaking turn of one track's rows from its raw pcm; (embeddings, their turns)."""
    pcm = np.memmap(raw, dtype=np.int16, mode="r", shape=(raw.stat().st_size // 2,))   # explicit: a torn odd last byte is ignored
    embeddings, turns = [], []
    for t in _merge_turns(rows):
        a, b = int(t["t_start"] * RATE), min(int(t["t_end"] * RATE), len(pcm))
        if b <= a:
            continue
        emb = embedder.embed(pcm[a:b].astype(np.float32) / 32768.0)
        if emb is not None:
            embeddings.append(emb)
            turns.append(t)
    return embeddings, turns


def _journal_key(j: Path) -> Optional[Tuple[int, int]]:
    try:
        st = j.stat()
        return st.st_mtime_ns, st.st_size
    except OSError:                    # mid-swap (refine replaces the journal) or gone
        return None


def identify_speakers(meeting_dir: Path) -> Optional[List[str]]:
    """Post-meeting pass: embeds each speaking turn (mic and loopback tracks), matches/clusters it against saved
    voice profiles, writes <meeting>/speakers.json (seq -> profile id) and updates the global
    profile store (existing centroids only on a meeting's first pass: meta.json voiceid_done). Returns the
    ids of speakers in this meeting that still have no name (the user names them from Settings > Voices, never by popup). No-ops quietly if the speaker-id model
    isn't installed. Returns None, having saved nothing, if the journal was replaced (refine) while
    it ran - its seq numbers no longer match, so the caller should run it again."""
    from . import export, store  # local import: store.read_rows() calls back into this module

    embedder = get_embedder()
    if embedder is None:
        return []
    journal = meeting_dir / "journal.jsonl"
    key = _journal_key(journal)
    rows = export.read_journal(journal)
    embeddings, turns_with_emb = [], []
    for track in ("mic", "loop"):      # turns never merge across tracks; both pool into one assignment below
        raw = meeting_dir / "raw" / f"{track}.pcm"
        track_rows = [r for r in rows if r["track"] == track]
        if not track_rows or not raw.exists() or raw.stat().st_size < 2:      # memmap rejects an empty file
            continue
        e, t = _track_embeddings(embedder, track_rows, raw)
        embeddings += e
        turns_with_emb += t
    if not embeddings:
        return [] if _journal_key(journal) == key else None
    learn = not store.read_meta(meeting_dir).get("voiceid_done")    # this audio is already in the centroids (refine/retranscribe)
    profiles = load_profiles()
    assigned, updated = assign_speakers(embeddings, profiles, learn=learn)   # outside _LOCK: the GUI thread takes it
    with _LOCK:      # only the read-modify-write is serialised; redo if Settings changed the store meanwhile
        current = load_profiles()
        if current != profiles:
            assigned, updated = assign_speakers(embeddings, current, learn=learn)
        if _journal_key(journal) != key:
            return None
        speakers: Dict[int, str] = {}
        for turn, pid in zip(turns_with_emb, assigned):
            for seq in turn["seqs"]:
                speakers[seq] = pid
        save_profiles(updated)
        if learn:
            store.update_meta(meeting_dir, voiceid_done=True)
        store.write_speakers(meeting_dir, speakers)
    named_ids = {p["id"] for p in updated if p.get("name")}
    return sorted({pid for pid in assigned if pid not in named_ids})
