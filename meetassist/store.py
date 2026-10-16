"""Meeting folders on disk: %LOCALAPPDATA%\\MeetAssist\\meetings\\<id>\\{meta.json, raw/, journal.jsonl, exports/}."""
from __future__ import annotations

import json
import os
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np

from . import paths
from .segmenter import RATE


def new_meeting(title: str = "", kind: str = "live", model: str = "", language: str = "mixed") -> Path:
    paths.ensure_dirs()
    now = time.time()
    mid = time.strftime("%Y%m%d-%H%M%S", time.localtime(now))
    d = paths.MEETINGS_DIR / mid
    n = 1
    while d.exists():
        n += 1; d = paths.MEETINGS_DIR / f"{mid}-{n}"
    (d / "raw").mkdir(parents=True); (d / "exports").mkdir()
    meta = {"id": d.name, "title": title or time.strftime("Meeting %d %b %Y %H:%M", time.localtime(now)),
            "kind": kind, "epoch": now, "started_at": time.strftime("%Y-%m-%d %H:%M", time.localtime(now)),
            "model": model, "language": language, "ended_at": None}
    write_meta(d, meta)
    return d


def read_meta(d: Path) -> dict:
    try:
        return json.loads((d / "meta.json").read_text(encoding="utf-8"))
    except Exception:
        return {"id": d.name, "title": d.name}


def write_meta(d: Path, meta: dict) -> None:
    tmp = d / "meta.tmp"
    tmp.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, d / "meta.json")


def update_meta(d: Path, **kw) -> dict:
    m = read_meta(d); m.update(kw); write_meta(d, m); return m


def read_json(p: Path) -> Optional[dict]:
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


@dataclass
class MeetingInfo:
    path: Path
    meta: dict

    @property
    def id(self) -> str: return self.path.name
    @property
    def title(self) -> str: return self.meta.get("title", self.path.name)


def list_meetings() -> List[MeetingInfo]:
    paths.ensure_dirs()
    out = [MeetingInfo(p, read_meta(p)) for p in paths.MEETINGS_DIR.iterdir() if (p / "meta.json").exists()]
    out.sort(key=lambda m: m.meta.get("epoch", 0), reverse=True)
    return out


def delete_meeting(d: Path) -> None:
    # AGENTS rule 2: raw audio is only ever removed on an explicit user action (this function).
    if d.resolve().parent == paths.MEETINGS_DIR.resolve():
        shutil.rmtree(d, ignore_errors=True)


def audio_seconds(d: Path) -> float:
    return max((((d / "raw" / f"{t}.pcm").stat().st_size / 2 / RATE) if (d / "raw" / f"{t}.pcm").exists() else 0.0)
               for t in ("mic", "loop"))


def decode_to_pcm(src: str, out_pcm: Path, progress=None) -> float:
    """Decode any audio/video file (mp3/m4a/wav/mp4/...) to 16 kHz mono int16 raw PCM. Returns seconds."""
    import av
    c = av.open(src)
    stream = c.streams.audio[0]
    total = float(stream.duration * stream.time_base) if stream.duration else (c.duration / 1e6 if c.duration else 0)
    r = av.AudioResampler(format="s16", layout="mono", rate=RATE)
    n = 0
    with open(out_pcm, "wb") as f:
        for frame in c.decode(stream):
            for x in r.resample(frame):
                b = bytes(x.planes[0])[: x.samples * 2]
                f.write(b); n += x.samples
            if progress and total:
                progress(min(n / RATE / total, 1.0))
        for x in r.resample(None) or []:
            b = bytes(x.planes[0])[: x.samples * 2]; f.write(b); n += x.samples
    c.close()
    return n / RATE
