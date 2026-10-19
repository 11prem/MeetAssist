"""Journal -> speaker-attributed transcript files (.txt .md .srt) with raw/cleaned/both variants."""
from __future__ import annotations

import bisect
import difflib
import json
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from . import tamil

LABELS = {"mic": "You", "loop": "Remote", "file": "Speaker"}


def read_journal(path: Path, hide_echo: bool = True) -> List[dict]:
    """Read fsynced JSONL; a torn trailing line (hard kill) is skipped. Speaker-bleed echo is hidden."""
    rows: List[dict] = []
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    rows.sort(key=lambda r: (r.get("t_start", 0), r.get("seq", 0)))
    if hide_echo:
        rows = [r for r in rows if not r.get("echo")]
        return drop_echo(rows)
    return rows


def _words(t: str) -> List[str]:
    return re.findall(r"[\w']+", t.lower())


def drop_echo(rows: List[dict], window: float = 3.0, min_ratio: float = 0.55) -> List[dict]:
    """Speaker-bleed removal: when you use laptop speakers your microphone also records the remote voices,
    so the same words appear on both tracks. A 'mic' piece whose words largely match overlapping
    'loop' pieces (within `window` s) is the echo -> hidden. The journal file itself is never modified."""
    loops = sorted((r for r in rows if r["track"] == "loop"), key=lambda r: r["t_start"])
    if not loops:
        return list(rows)
    starts = [l["t_start"] for l in loops]
    out: List[dict] = []
    for r in rows:
        if r["track"] == "mic":
            w = _words(r.get("text", ""))
            if len(w) >= 3:
                lo = bisect.bisect_left(starts, r["t_start"] - window - 35.0)      # pieces never span > ~35 s
                hi = bisect.bisect_right(starts, r["t_end"] + window)
                near = [l for l in loops[lo:hi] if l["t_end"] >= r["t_start"] - window]
                if near:
                    lw = _words(" ".join(l.get("text", "") for l in near))
                    if lw:
                        sm = difflib.SequenceMatcher(None, w, lw, autojunk=False)
                        if sum(b.size for b in sm.get_matching_blocks()) / len(w) >= min_ratio:
                            continue
        out.append(r)
    return out


def hms(sec: float) -> str:
    s = int(max(sec, 0)); return f"{s // 3600:02d}:{s % 3600 // 60:02d}:{s % 60:02d}"


def srt_ts(sec: float) -> str:
    ms = int(round(max(sec, 0) * 1000)); h, ms = divmod(ms, 3600000); m, ms = divmod(ms, 60000); s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def _text(r: dict, variant: str, cleaned: Optional[Dict[int, str]], disp=False) -> str:
    """disp: 'english' (Tamil pieces as their English translation) | 'roman' (Tamil in English letters) |
    'tamil' (Tamil script). Legacy bool: True = roman, False = tamil."""
    mode = "roman" if disp is True else "tamil" if disp is False else disp
    if variant == "cleaned" and cleaned and r.get("seq") in cleaned:
        t = cleaned[r["seq"]]
    elif mode == "english" and r.get("en"):
        t = r["en"]
    else:
        t = r.get("text", "")
    return tamil.romanize(t) if mode in ("roman", "english") else t


def attach_speakers(rows: List[dict], speakers: Dict[int, str], names: Dict[str, str]) -> List[dict]:
    """Add a 'who' field (resolved display name) to rows with a named, recognised voice. Rows with
    no recognised/named speaker are untouched and keep falling back to the mic/loop track label."""
    for r in rows:
        name = names.get(speakers.get(r.get("seq")))
        if name:
            r["who"] = name
    return rows


def group_rows(rows: Iterable[dict], variant="raw", cleaned=None, max_gap=4.0, max_chars=700, roman=False) -> List[dict]:
    """Merge consecutive same-speaker pieces (gap <= max_gap s) into paragraphs for reading."""
    groups: List[dict] = []
    for r in rows:
        t = _text(r, variant, cleaned, roman).strip()
        if not t:
            continue
        who = r.get("who")
        g = groups[-1] if groups else None
        if (g and g["track"] == r["track"] and g.get("who") == who and (roman == "english" or g.get("lang") == r.get("lang"))
                and r["t_start"] - g["t_end"] <= max_gap and len(g["text"]) < max_chars):
            g["text"] += " " + t
            g["t_end"] = max(g["t_end"], r["t_end"])
        else:
            groups.append({"track": r["track"], "who": who, "t_start": r["t_start"], "t_end": r["t_end"], "text": t, "lang": r.get("lang")})
    return groups


def render_txt(rows: Iterable[dict], variant="raw", cleaned=None, labels=LABELS, roman=False) -> str:
    out = [f"[{hms(g['t_start'])}] [{g['who'] or labels.get(g['track'], g['track'])}] {g['text']}"
           for g in group_rows(rows, variant, cleaned, roman=roman)]
    return "\n".join(out) + "\n"


def render_srt(rows: Iterable[dict], variant="raw", cleaned=None, labels=LABELS, roman=False) -> str:
    out, n = [], 0
    for r in rows:
        t = _text(r, variant, cleaned, roman).strip()
        if not t:
            continue
        n += 1
        out.append(f"{n}\n{srt_ts(r['t_start'])} --> {srt_ts(max(r['t_end'], r['t_start'] + 0.5))}\n"
                   f"{r.get('who') or labels.get(r['track'], r['track'])}: {t}\n")
    return "\n".join(out)


def render_md(rows: List[dict], meta: dict, variant="raw", cleaned=None, summary: str = "", labels=LABELS, roman=False) -> str:
    dur = max((r["t_end"] for r in rows), default=0)
    head = [f"# {meta.get('title', 'Meeting')}", "",
            f"- **Date:** {meta.get('started_at', '')}", f"- **Duration:** {hms(dur)}",
            f"- **Model:** {meta.get('model', '')}", ""]
    if summary.strip():
        head += [summary.strip(), ""]
    head += ["## Transcript", ""]
    body = [f"**[{hms(g['t_start'])}] {g['who'] or labels.get(g['track'], g['track'])}:** {g['text']}\n"
            for g in group_rows(rows, variant, cleaned, roman=roman)]
    return "\n".join(head + body) + "\n"
