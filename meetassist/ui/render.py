"""Rich-text rendering of the transcript / summary in the active theme (reading-optimised: 16 px, 155 % leading)."""
from __future__ import annotations

import html
from typing import List

from PySide6.QtCore import Qt
from PySide6.QtGui import QFont
from PySide6.QtWidgets import QTextBrowser

from .. import export
from . import theme
from .theme import READ_WIDTH

SPEAKER = {"mic": "You", "loop": "Remote", "file": "Speaker"}


def transcript_html(groups: List[dict], p: theme.Palette) -> str:
    out = []
    for g in groups:
        col = p.you if g["track"] == "mic" else p.remote if g["track"] == "loop" else p.text2
        who = SPEAKER.get(g["track"], g["track"])
        out.append(
            f"<div style='margin:0 0 20px 0'>"
            f"<p style='margin:0 0 3px 0'><span style='color:{col};font-weight:600;font-size:13px'>{who}</span>"
            f"<span style='color:{p.text3};font-size:12px'>&nbsp;&nbsp;{export.hms(g['t_start'])}</span></p>"
            f"<p style='margin:0;line-height:155%;font-size:16px;color:{p.text}'>{html.escape(g['text'])}</p></div>")
    return "".join(out)


def placeholder_html(title: str, body: str, p: theme.Palette) -> str:
    return (f"<div style='margin-top:36px'><p style='margin:0 0 6px 0;font-size:17px;font-weight:600;color:{p.text2}'>{html.escape(title)}</p>"
            f"<p style='margin:0;font-size:14px;line-height:150%;color:{p.text3}'>{html.escape(body)}</p></div>")


def summary_css(p: theme.Palette) -> str:
    return (f"body {{ color:{p.text}; font-size:15px; }} h1,h2,h3 {{ color:{p.text}; font-weight:700; }} "
            f"h2 {{ font-size:19px; margin-top:22px; }} li {{ line-height:150%; margin-bottom:4px; }} a {{ color:{p.accent}; }} "
            f"p {{ line-height:150%; }}")


class ReadingView(QTextBrowser):
    """QTextBrowser that keeps a comfortable reading measure (centred column, max READ_WIDTH)."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setFrameShape(QTextBrowser.Shape.NoFrame)
        self.setOpenLinks(False)
        f = QFont(); f.setFamilies(["Segoe UI Variable Text", "Segoe UI", "Nirmala UI"]); f.setPixelSize(16)
        self.setFont(f)
        self.document().setDocumentMargin(0)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

    def resizeEvent(self, e) -> None:
        side = max(28, (self.width() - READ_WIDTH) // 2)
        self.setViewportMargins(side, 12, side, 12)       # margins first, so the document lays out at the final width
        super().resizeEvent(e)
        self.document().setTextWidth(self.viewport().width())


def chat_html(history: List[dict], p: theme.Palette) -> str:
    out = []
    for m in history:
        you = m.get("role") == "user"
        who, col = ("You", p.you) if you else ("Assistant", p.accent)
        out.append(
            f"<p style='margin:0 0 3px 0'><span style='color:{col};font-weight:600;font-size:13px'>{who}</span></p>"
            f"<p style='margin:0 0 18px 0;line-height:155%;font-size:15px;color:{p.text}'>{html.escape(m.get('content', ''))}</p>")
    return "".join(out)


def markdown_html(md: str, p: theme.Palette) -> str:
    """Tiny Markdown subset (##, bullets, checkboxes, **bold**) -> themed HTML for the summary tab."""
    import re

    def inline(t: str) -> str:
        t = html.escape(t)
        return re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", t)

    out, in_list = [], False
    for raw in md.splitlines():
        ln = raw.rstrip()
        if not ln.strip():
            continue
        m = re.match(r"^\s*[-*]\s+\[( |x|X)\]\s+(.*)", ln)
        if m:
            if in_list: out.append("</ul>"); in_list = False
            done = m.group(1).lower() == "x"
            box = f"<span style='color:{p.accent if not done else p.success};font-size:17px'>{'&#9745;' if done else '&#9744;'}</span>"
            out.append(f"<p style='margin:0 0 9px 2px;line-height:150%;font-size:15px;color:{p.text}'>{box}&nbsp;&nbsp;{inline(m.group(2))}</p>")
            continue
        m = re.match(r"^\s*[-*]\s+(.*)", ln)
        if m:
            if not in_list: out.append("<ul style='margin:0 0 4px 18px'>"); in_list = True
            out.append(f"<li style='margin-bottom:2px;line-height:150%;font-size:15px;color:{p.text}'>{inline(m.group(1))}</li>")
            continue
        if in_list: out.append("</ul>"); in_list = False
        h = re.match(r"^#{1,6}\s+(.*)", ln)
        if h:
            out.append(f"<p style='margin:26px 0 10px 0;font-size:19px;font-weight:700;color:{p.text}'>{inline(h.group(1))}</p>")
        else:
            out.append(f"<p style='margin:0 0 10px 0;line-height:150%;font-size:15px;color:{p.text2}'>{inline(ln)}</p>")
    if in_list:
        out.append("</ul>")
    return "<div style='margin-top:6px'>" + "".join(out) + "</div>"
