"""MeetAssist app mark, drawn in code (vector, crisp at every size, nothing to lose).

Concept: a premium squircle (iOS-style superellipse) in an electric-blue -> indigo -> violet gradient with a soft light
bloom and glass highlight. The symbol is five rounded bars that read as BOTH a sound waveform and the letter "M"
(tall outer strokes, descending to a valley); the centre bar is the red "recording" dot. Small sizes get bolder bars.
"""
from __future__ import annotations

import math

from PySide6.QtCore import QBuffer, QByteArray, QIODevice, QPointF, QRectF, Qt
from PySide6.QtGui import (QBrush, QColor, QIcon, QImage, QLinearGradient, QPainter, QPainterPath, QPen, QPixmap,
                           QRadialGradient)

HEIGHTS = (0.56, 0.36, 0.20, 0.36, 0.56)      # the "M" valley; centre bar (index 2) is the record dot
ICO_SIZES = (16, 20, 24, 32, 40, 48, 64, 96, 128, 256)


def _squircle(rect: QRectF, n: float = 5.0, steps: int = 240) -> QPainterPath:
    cx, cy, a, b = rect.center().x(), rect.center().y(), rect.width() / 2, rect.height() / 2
    path = QPainterPath()
    for i in range(steps + 1):
        t = 2 * math.pi * i / steps
        c, s = math.cos(t), math.sin(t)
        x = cx + a * math.copysign(abs(c) ** (2 / n), c)
        y = cy + b * math.copysign(abs(s) ** (2 / n), s)
        path.moveTo(x, y) if i == 0 else path.lineTo(x, y)
    path.closeSubpath()
    return path


def paint_mark(g: QPainter, S: float) -> None:
    g.setRenderHint(QPainter.RenderHint.Antialiasing)
    pad = S * 0.04
    box = QRectF(pad, pad, S - 2 * pad, S - 2 * pad)
    sq = _squircle(box)

    base = QLinearGradient(box.topLeft(), box.bottomRight())
    base.setColorAt(0.0, QColor("#38BDF8")); base.setColorAt(0.45, QColor("#3B6CF6")); base.setColorAt(1.0, QColor("#6D28D9"))
    g.setPen(Qt.PenStyle.NoPen); g.setBrush(QBrush(base)); g.drawPath(sq)

    bloom = QRadialGradient(QPointF(box.left() + box.width() * 0.25, box.top() + box.height() * 0.15), box.width() * 0.75)
    bloom.setColorAt(0, QColor(255, 255, 255, 92)); bloom.setColorAt(1, QColor(255, 255, 255, 0))
    g.setBrush(QBrush(bloom)); g.drawPath(sq)

    g.save(); g.setClipPath(sq)                                              # glass highlight on the upper half
    hl = QLinearGradient(box.topLeft(), QPointF(box.left(), box.top() + box.height() * 0.55))
    hl.setColorAt(0, QColor(255, 255, 255, 48)); hl.setColorAt(1, QColor(255, 255, 255, 0))
    g.setBrush(QBrush(hl)); g.drawRect(box.adjusted(0, 0, 0, -box.height() * 0.45))
    sh = QLinearGradient(QPointF(box.left(), box.bottom() - box.height() * 0.35), box.bottomLeft())   # gentle depth at the bottom
    sh.setColorAt(0, QColor(30, 10, 90, 0)); sh.setColorAt(1, QColor(30, 10, 90, 70))
    g.setBrush(QBrush(sh)); g.drawRect(box)
    g.restore()

    g.setBrush(Qt.BrushStyle.NoBrush); g.setPen(QPen(QColor(255, 255, 255, 64), max(1.0, S * 0.006)))   # inner hairline
    g.drawPath(sq)

    small = S <= 40
    bw = S * (0.115 if small else 0.088)
    gap = S * (0.055 if small else 0.052)
    total = len(HEIGHTS) * bw + (len(HEIGHTS) - 1) * gap
    x0, cy = (S - total) / 2, S / 2
    if not small:                                                             # soft red glow behind the record dot (under all bars)
        cx = x0 + 2 * (bw + gap) + bw / 2
        glow = QRadialGradient(QPointF(cx, cy), bw * 2.6)
        glow.setColorAt(0, QColor(255, 80, 60, 110)); glow.setColorAt(1, QColor(255, 80, 60, 0))
        g.setPen(Qt.PenStyle.NoPen); g.setBrush(QBrush(glow)); g.drawEllipse(QPointF(cx, cy), bw * 2.6, bw * 2.6)
    for i, h in enumerate(HEIGHTS):
        hh = S * h * (1.06 if small else 1.0)
        r = QRectF(x0 + i * (bw + gap), cy - hh / 2, bw, hh)
        centre = i == 2
        g.setPen(Qt.PenStyle.NoPen)
        g.setBrush(QBrush(QColor(20, 10, 80, 72)))                           # contact shadow
        g.drawRoundedRect(r.translated(0, S * 0.014), bw / 2, bw / 2)
        fill = QLinearGradient(r.topLeft(), r.bottomLeft())
        if centre:
            fill.setColorAt(0, QColor("#FF8A7A")); fill.setColorAt(1, QColor("#FF3B30"))
        else:
            fill.setColorAt(0, QColor("#FFFFFF")); fill.setColorAt(1, QColor("#DCE6FF"))
        g.setBrush(QBrush(fill)); g.drawRoundedRect(r, bw / 2, bw / 2)


def make_image(size: int) -> QImage:
    img = QImage(size, size, QImage.Format.Format_ARGB32)
    img.fill(Qt.GlobalColor.transparent)
    g = QPainter(img)
    paint_mark(g, float(size))
    g.end()
    return img


def make_pixmap(size: int = 256) -> QPixmap:
    return QPixmap.fromImage(make_image(size))


def make_icon() -> QIcon:
    ic = QIcon()
    for s in (16, 20, 24, 32, 48, 64, 128, 256):
        ic.addPixmap(make_pixmap(s))
    return ic


def ico_bytes() -> bytes:
    """Multi-size Windows .ico (PNG-compressed entries) so the exe / taskbar / alt-tab icons are crisp at every size."""
    entries = []
    for s in ICO_SIZES:
        buf = QBuffer(); buf.open(QIODevice.OpenModeFlag.WriteOnly)
        make_image(s).save(buf, "PNG")
        entries.append((s, bytes(buf.data())))
    head = bytearray(b"\x00\x00\x01\x00" + len(entries).to_bytes(2, "little"))
    offset = 6 + 16 * len(entries)
    body = bytearray()
    for s, data in entries:
        head += bytes([s if s < 256 else 0, s if s < 256 else 0, 0, 0]) + (1).to_bytes(2, "little") + (32).to_bytes(2, "little")
        head += len(data).to_bytes(4, "little") + offset.to_bytes(4, "little")
        offset += len(data); body += data
    return bytes(head + body)
