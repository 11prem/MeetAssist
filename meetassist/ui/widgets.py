"""Custom-painted widgets for the Apple-style UI. Every colour comes from theme.P(); every animation uses MOTION tokens."""
from __future__ import annotations

import math
from collections import deque
from typing import List, Optional

from PySide6.QtCore import (QEasingCurve, QPointF, QPropertyAnimation, QRectF, QSize, Qt, QTimer, QVariantAnimation, Signal)
from PySide6.QtGui import (QBrush, QColor, QFont, QFontMetrics, QPainter, QPainterPath, QPen)
from PySide6.QtWidgets import (QAbstractButton, QFrame, QGraphicsDropShadowEffect, QGraphicsOpacityEffect, QHBoxLayout,
                               QLabel, QSizePolicy, QStyle, QStyledItemDelegate, QToolButton, QVBoxLayout, QWidget)

from . import icons, theme
from .theme import MOTION, RADIUS, SP, TYPE


def qcolor(c: str, alpha: float = 1.0) -> QColor:
    q = QColor(c)
    q.setAlphaF(alpha)
    return q


def label(text: str = "", role: str = "body2", wrap: bool = False) -> QLabel:
    lb = QLabel(text)
    lb.setProperty("role", role)
    lb.setWordWrap(wrap)
    return lb


def shadow(widget: QWidget, blur: int = 28, dy: int = 6) -> None:
    eff = QGraphicsDropShadowEffect(widget)
    eff.setBlurRadius(blur)
    eff.setOffset(0, dy)
    p = theme.P()
    eff.setColor(QColor(0, 0, 0, 70 if p.dark else 38))
    widget.setGraphicsEffect(eff)


def reduced_motion() -> bool:
    return False          # hook: honour Windows "show animations" later; all animations are short and optional


# ----------------------------------------------------------------------------- icon button
class IconButton(QToolButton):
    """32x32 (hit area 36) icon-only button; always carries a tooltip + accessible name."""

    def __init__(self, name: str, tip: str, size: int = theme.ICON["md"], accent: bool = False, parent=None) -> None:
        super().__init__(parent)
        self._name, self._size, self._accent = name, size, accent
        self.setToolTip(tip)
        self.setAccessibleName(tip)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setFixedSize(36, 36)
        self.setIconSize(QSize(size, size))
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        theme.init().changed.connect(self._paint_icon)
        self._paint_icon()

    def set_name(self, name: str) -> None:
        self._name = name
        self._paint_icon()

    def _paint_icon(self) -> None:
        p = theme.P()
        self.setIcon(icons.icon(self._name, p.accent if self._accent else p.text2, p.accent if self._accent else p.text, p.placeholder, self._size))


class LabelButton(QToolButton):
    """Icon + text toolbar button (e.g. language view menu, AI menu)."""

    def __init__(self, name: str, text: str, tip: str = "", parent=None) -> None:
        super().__init__(parent)
        self._name = name
        self.setText(text)
        self.setToolTip(tip or text)
        self.setProperty("labelled", True)
        self.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setIconSize(QSize(theme.ICON["md"], theme.ICON["md"]))
        self.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        theme.init().changed.connect(self._paint_icon)
        self._paint_icon()

    def _paint_icon(self) -> None:
        p = theme.P()
        self.setIcon(icons.icon(self._name, p.text2, p.text, p.placeholder, theme.ICON["md"]))


class SideButton(QAbstractButton):
    """Sidebar row: icon + label, 36px high."""

    def __init__(self, name: str, text: str, parent=None) -> None:
        super().__init__(parent)
        self._name = name
        self.setText(text)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setFixedHeight(36)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self._hover = False
        theme.init().changed.connect(self.update)

    def enterEvent(self, e):
        self._hover = True; self.update()

    def leaveEvent(self, e):
        self._hover = False; self.update()

    def sizeHint(self) -> QSize:
        return QSize(200, 36)

    def paintEvent(self, e) -> None:
        p = theme.P()
        g = QPainter(self)
        g.setRenderHint(QPainter.RenderHint.Antialiasing)
        r = QRectF(self.rect()).adjusted(0.5, 0.5, -0.5, -0.5)
        if self._hover or self.isDown():
            g.setBrush(QBrush(qcolor(p.fill_hover if self.isDown() else p.fill)))
            g.setPen(Qt.PenStyle.NoPen)
            g.drawRoundedRect(r, RADIUS["md"], RADIUS["md"])
        if self.hasFocus():
            g.setBrush(Qt.BrushStyle.NoBrush)
            g.setPen(QPen(qcolor(p.accent), 2))
            g.drawRoundedRect(r.adjusted(1, 1, -1, -1), RADIUS["md"], RADIUS["md"])
        col = p.text if self._hover else p.text2
        g.drawPixmap(12, (self.height() - theme.ICON["md"]) // 2, icons.pixmap(self._name, col, theme.ICON["md"]))
        f = self.font(); f.setPixelSize(TYPE["body"]); f.setWeight(QFont.Weight.Medium)
        g.setFont(f); g.setPen(qcolor(col))
        g.drawText(QRectF(12 + theme.ICON["md"] + 10, 0, self.width() - 52, self.height()), Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft, self.text())


# ----------------------------------------------------------------------------- record button (the hero control)
class RecordButton(QAbstractButton):
    """Red disc in a ring (Voice-Memos style). Recording -> rounded stop square with a soft expanding pulse."""

    def __init__(self, parent=None, size: int = 56) -> None:
        super().__init__(parent)
        self.setFixedSize(size, size)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setAccessibleName("Start recording")
        self.setToolTip("Start recording  (Ctrl+R)")
        self._rec = False
        self._morph = 0.0              # 0 = disc, 1 = stop square
        self._pulse = 0.0
        self._hover = 0.0
        self._busy = False
        self._m = QVariantAnimation(self, duration=MOTION["base"], easingCurve=QEasingCurve.Type.OutCubic)
        self._m.valueChanged.connect(lambda v: (setattr(self, "_morph", v), self.update()))
        self._h = QVariantAnimation(self, duration=MOTION["fast"], easingCurve=QEasingCurve.Type.OutCubic)
        self._h.valueChanged.connect(lambda v: (setattr(self, "_hover", v), self.update()))
        self._pl = QVariantAnimation(self, duration=MOTION["pulse"], startValue=0.0, endValue=1.0, loopCount=-1)
        self._pl.valueChanged.connect(lambda v: (setattr(self, "_pulse", v), self.update()))
        theme.init().changed.connect(self.update)

    def set_recording(self, on: bool) -> None:
        if on == self._rec:
            return
        self._rec = on
        self.setAccessibleName("Stop recording" if on else "Start recording")
        self.setToolTip("Stop recording  (Ctrl+R)" if on else "Start recording  (Ctrl+R)")
        self._m.stop(); self._m.setStartValue(self._morph); self._m.setEndValue(1.0 if on else 0.0); self._m.start()
        if on:
            self._pl.start()
        else:
            self._pl.stop(); self._pulse = 0.0

    def set_busy(self, on: bool) -> None:
        self._busy = on
        self.setEnabled(not on)
        self.update()

    def enterEvent(self, e):
        self._h.stop(); self._h.setStartValue(self._hover); self._h.setEndValue(1.0); self._h.start()

    def leaveEvent(self, e):
        self._h.stop(); self._h.setStartValue(self._hover); self._h.setEndValue(0.0); self._h.start()

    def paintEvent(self, e) -> None:
        p = theme.P()
        g = QPainter(self)
        g.setRenderHint(QPainter.RenderHint.Antialiasing)
        w = self.width(); c = QPointF(w / 2, w / 2)
        if self._rec:                                                          # expanding pulse ring
            for k in (0.0, 0.5):
                t = (self._pulse + k) % 1.0
                g.setPen(Qt.PenStyle.NoPen)
                g.setBrush(QBrush(qcolor(p.record, 0.28 * (1 - t))))
                rr = w * 0.5 * (0.72 + 0.28 * t) + 0.0
                g.drawEllipse(c, rr, rr)
        ring = w / 2 - 2
        g.setBrush(Qt.BrushStyle.NoBrush)
        g.setPen(QPen(qcolor(p.text3 if not self._rec else p.record, 0.9), 2.5))
        g.drawEllipse(c, ring, ring)
        # inner shape: disc (radius ~ 0.36w) -> rounded square (side ~ 0.34w)
        half = w * (0.36 - 0.17 * self._morph) + 1.0 * self._hover * (1 - self._morph)
        radius = half * (1.0 - 0.72 * self._morph)
        col = QColor(p.record_hover if self._hover > 0.5 else p.record)
        if self._busy:
            col.setAlphaF(0.5)
        g.setPen(Qt.PenStyle.NoPen)
        g.setBrush(QBrush(col))
        g.drawRoundedRect(QRectF(c.x() - half, c.y() - half, half * 2, half * 2), radius, radius)
        if self.hasFocus():
            g.setBrush(Qt.BrushStyle.NoBrush)
            g.setPen(QPen(qcolor(p.accent), 2))
            g.drawEllipse(c, w / 2 - 0.5, w / 2 - 0.5)

    def keyPressEvent(self, e) -> None:
        if e.key() in (Qt.Key.Key_Space, Qt.Key.Key_Return, Qt.Key.Key_Enter):
            self.click()
        else:
            super().keyPressEvent(e)


# ----------------------------------------------------------------------------- live waveform + meters
class WaveStrip(QWidget):
    """Two-lane live waveform: microphone (You) rises from the centre line, system audio (Remote) falls below it."""
    BARS = 96

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.mic: deque = deque([0.0] * self.BARS, maxlen=self.BARS)
        self.loop: deque = deque([0.0] * self.BARS, maxlen=self.BARS)
        self.setMinimumHeight(44)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.setFixedHeight(48)
        self._active = False
        theme.init().changed.connect(self.update)

    def set_active(self, on: bool) -> None:
        self._active = on
        if not on:
            self.mic.extend([0.0] * self.BARS); self.loop.extend([0.0] * self.BARS)
        self.update()

    def push(self, mic: float, loop: float) -> None:
        self.mic.append(max(0.0, min(1.0, mic))); self.loop.append(max(0.0, min(1.0, loop)))
        self.update()

    def paintEvent(self, e) -> None:
        p = theme.P()
        g = QPainter(self)
        g.setRenderHint(QPainter.RenderHint.Antialiasing)
        w, h = self.width(), self.height()
        n = min(self.BARS, max(8, int(w // 6)))
        gap = w / n
        bw = max(2.0, gap * 0.5)
        mid = h / 2
        g.setPen(Qt.PenStyle.NoPen)
        mics, loops = list(self.mic)[-n:], list(self.loop)[-n:]
        for i in range(n):
            x = i * gap + (gap - bw) / 2
            age = (i + 1) / n                                                    # newer bars are stronger
            for vals, col, up in ((mics, p.you, True), (loops, p.remote, False)):
                v = vals[i] if i < len(vals) else 0.0
                hh = max(1.5, (mid - 2) * (v ** 0.7)) if self._active else 1.5
                a = (0.35 + 0.65 * age) if self._active else 0.35
                g.setBrush(QBrush(qcolor(col, a)))
                y = mid - hh if up else mid
                g.drawRoundedRect(QRectF(x, y, bw, hh if hh > 1.6 else 1.6), bw / 2, bw / 2)


class LevelMeter(QWidget):
    """Thin pill meter with attack/release smoothing. value in 0..1."""

    def __init__(self, color_role: str = "you", parent=None) -> None:
        super().__init__(parent)
        self._role, self._target, self._shown = color_role, 0.0, 0.0
        self.setFixedHeight(6)
        self.setMinimumWidth(72)
        self._t = QTimer(self, interval=33, timeout=self._step)
        theme.init().changed.connect(self.update)

    def set_value(self, v: float) -> None:
        self._target = max(0.0, min(1.0, v))
        if not self._t.isActive():
            self._t.start()

    def _step(self) -> None:
        d = self._target - self._shown
        self._shown += d * (0.5 if d > 0 else 0.12)
        if abs(d) < 0.004 and self._target == 0.0:
            self._shown = 0.0; self._t.stop()
        self.update()

    def paintEvent(self, e) -> None:
        p = theme.P()
        g = QPainter(self)
        g.setRenderHint(QPainter.RenderHint.Antialiasing)
        r = QRectF(self.rect())
        g.setPen(Qt.PenStyle.NoPen)
        g.setBrush(QBrush(qcolor(p.fill)))
        g.drawRoundedRect(r, 3, 3)
        if self._shown > 0.01:
            g.setBrush(QBrush(qcolor(getattr(p, self._role))))
            g.drawRoundedRect(QRectF(0, 0, max(6.0, r.width() * self._shown), r.height()), 3, 3)


class MeterRow(QWidget):
    """[icon] [name] [meter] - one per audio source, with an accessible name. Click to mute/unmute that source."""

    clicked = Signal()

    def __init__(self, icon_name: str, role: str, name: str, parent=None) -> None:
        super().__init__(parent)
        self._icon, self._role, self._name_full, self._muted = icon_name, role, name, False
        self.setAccessibleName(name + " level")
        self.setToolTip(f"Click to mute {name}")
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        lay = QHBoxLayout(self); lay.setContentsMargins(0, 0, 0, 0); lay.setSpacing(SP["sm"])
        self.ic = QLabel(); self.ic.setFixedSize(theme.ICON["md"], theme.ICON["md"])
        self.name = label(name.split(" ")[0], "caption"); self.name.setFixedWidth(46)
        self.meter = LevelMeter(role)
        lay.addWidget(self.ic); lay.addWidget(self.name); lay.addWidget(self.meter, 1)
        theme.init().changed.connect(self._paint)
        self._paint()

    def set_muted(self, muted: bool) -> None:
        if muted == self._muted:
            return
        self._muted = muted
        self.setToolTip(f"Click to {'enable' if muted else 'mute'} {self._name_full}")
        self.setAccessibleName(self._name_full + " level" + (" (muted)" if muted else ""))
        self._paint()

    def _paint(self) -> None:
        p = theme.P()
        col = p.text3 if self._muted else getattr(p, self._role)
        self.ic.setPixmap(icons.pixmap(self._icon, col, theme.ICON["md"]))
        self.name.setEnabled(not self._muted)

    def set_value(self, v: float) -> None:
        self.meter.set_value(v)

    def mousePressEvent(self, e) -> None:
        if e.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit()
        else:
            super().mousePressEvent(e)


class SourceStatusLabel(QWidget):
    """"Microphone + system audio" - the disabled source's word is struck through, wiped on left to right."""

    def __init__(self, mic_on: bool = True, sys_on: bool = True, parent=None) -> None:
        super().__init__(parent)
        self._mic_on, self._sys_on = mic_on, sys_on
        self._mic_strike, self._sys_strike = (0.0 if mic_on else 1.0), (0.0 if sys_on else 1.0)
        f = QFont(); f.setPixelSize(TYPE["footnote"])
        self.setFont(f)
        self.setFixedHeight(QFontMetrics(f).height())
        self._mic_a = QVariantAnimation(self, duration=MOTION["slow"], easingCurve=QEasingCurve.Type.OutCubic)
        self._mic_a.valueChanged.connect(lambda v: (setattr(self, "_mic_strike", v), self.update()))
        self._sys_a = QVariantAnimation(self, duration=MOTION["slow"], easingCurve=QEasingCurve.Type.OutCubic)
        self._sys_a.valueChanged.connect(lambda v: (setattr(self, "_sys_strike", v), self.update()))
        theme.init().changed.connect(self.update)

    def set_sources(self, mic_on: bool, sys_on: bool) -> None:
        if mic_on != self._mic_on:
            self._mic_on = mic_on
            self._mic_a.stop(); self._mic_a.setStartValue(self._mic_strike); self._mic_a.setEndValue(0.0 if mic_on else 1.0); self._mic_a.start()
        if sys_on != self._sys_on:
            self._sys_on = sys_on
            self._sys_a.stop(); self._sys_a.setStartValue(self._sys_strike); self._sys_a.setEndValue(0.0 if sys_on else 1.0); self._sys_a.start()

    def sizeHint(self) -> QSize:
        fm = QFontMetrics(self.font())
        return QSize(fm.horizontalAdvance("Microphone + system audio") + 2, fm.height())

    def paintEvent(self, e) -> None:
        p = theme.P()
        g = QPainter(self)
        g.setRenderHint(QPainter.RenderHint.Antialiasing)
        fm = QFontMetrics(self.font())
        g.setFont(self.font())
        x, cy = 0.0, self.height() / 2
        for text, strike in (("Microphone", self._mic_strike), (" + ", None), ("system audio", self._sys_strike)):
            w = fm.horizontalAdvance(text)
            g.setPen(qcolor(p.text3))
            g.drawText(QRectF(x, 0, w, self.height()), Qt.AlignmentFlag.AlignVCenter, text)
            if strike:
                g.setPen(QPen(qcolor(p.text3), 1.2))
                g.drawLine(QPointF(x, cy), QPointF(x + w * strike, cy))
            x += w


# ----------------------------------------------------------------------------- segmented control
class Segmented(QWidget):
    currentChanged = Signal(int)

    def __init__(self, labels: List[str], parent=None) -> None:
        super().__init__(parent)
        self._labels, self._i, self._x = labels, 0, 0.0
        self._enabled = [True] * len(labels)
        self.setFixedHeight(34)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self._anim = QVariantAnimation(self, duration=MOTION["base"], easingCurve=QEasingCurve.Type.OutCubic)
        self._anim.valueChanged.connect(lambda v: (setattr(self, "_x", v), self.update()))
        theme.init().changed.connect(self.update)
        f = QFont(self.font()); f.setPixelSize(TYPE["footnote"] + 1); f.setWeight(QFont.Weight.Medium)
        fm = QFontMetrics(f)
        self._wseg = max(fm.horizontalAdvance(t) for t in labels) + 36
        self.setFixedWidth(self._wseg * len(labels) + 6)

    def set_enabled(self, idx: int, on: bool) -> None:
        self._enabled[idx] = on
        self.update()

    def current(self) -> int:
        return self._i

    def set_current(self, i: int, animate: bool = True) -> None:
        if i == self._i:
            return
        self._i = i
        if animate:
            self._anim.stop(); self._anim.setStartValue(self._x); self._anim.setEndValue(float(i)); self._anim.start()
        else:
            self._x = float(i); self.update()
        self.currentChanged.emit(i)

    def mousePressEvent(self, e) -> None:
        i = int((e.position().x() - 3) // self._wseg)
        if 0 <= i < len(self._labels):
            self.set_current(i)

    def keyPressEvent(self, e) -> None:
        if e.key() in (Qt.Key.Key_Right, Qt.Key.Key_Down):
            self.set_current(min(self._i + 1, len(self._labels) - 1))
        elif e.key() in (Qt.Key.Key_Left, Qt.Key.Key_Up):
            self.set_current(max(self._i - 1, 0))
        else:
            super().keyPressEvent(e)

    def paintEvent(self, e) -> None:
        p = theme.P()
        g = QPainter(self)
        g.setRenderHint(QPainter.RenderHint.Antialiasing)
        r = QRectF(self.rect()).adjusted(0.5, 0.5, -0.5, -0.5)
        g.setPen(Qt.PenStyle.NoPen)
        g.setBrush(QBrush(qcolor(p.fill)))
        g.drawRoundedRect(r, RADIUS["md"], RADIUS["md"])
        sel = QRectF(3 + self._x * self._wseg, 3, self._wseg, self.height() - 6)
        g.setBrush(QBrush(qcolor(p.surface if not p.dark else p.fill_press)))
        g.setPen(QPen(qcolor(p.sep if not p.dark else p.border), 1))
        g.drawRoundedRect(sel, RADIUS["md"] - 2, RADIUS["md"] - 2)
        f = QFont(self.font()); f.setPixelSize(TYPE["footnote"] + 1); f.setWeight(QFont.Weight.Medium)
        g.setFont(f)
        for k, t in enumerate(self._labels):
            col = p.text if k == self._i else p.text3
            if not self._enabled[k]:
                col = p.placeholder
            g.setPen(qcolor(col))
            g.drawText(QRectF(3 + k * self._wseg, 3, self._wseg, self.height() - 6), Qt.AlignmentFlag.AlignCenter, t)
        if self.hasFocus():
            g.setBrush(Qt.BrushStyle.NoBrush)
            g.setPen(QPen(qcolor(p.accent), 2))
            g.drawRoundedRect(r.adjusted(1, 1, -1, -1), RADIUS["md"], RADIUS["md"])


# ----------------------------------------------------------------------------- status pill + banner + toast
class StatusPill(QWidget):
    """Coloured dot + text. kind: idle | live | working | done | warn"""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._kind, self._text, self._ph = "idle", "", 0.0
        self.setFixedHeight(22)
        self._a = QVariantAnimation(self, duration=MOTION["pulse"], startValue=0.0, endValue=1.0, loopCount=-1)
        self._a.valueChanged.connect(lambda v: (setattr(self, "_ph", v), self.update()))
        theme.init().changed.connect(self.update)

    def set(self, kind: str, text: str) -> None:
        if (kind, text) == (self._kind, self._text):
            return
        self._kind, self._text = kind, text
        f = QFont(self.font()); f.setPixelSize(TYPE["footnote"])
        self.setFixedWidth(QFontMetrics(f).horizontalAdvance(text) + 26 if text else 0)
        if kind in ("live", "working") and not self._a.state() == QVariantAnimation.State.Running:
            self._a.start()
        elif kind not in ("live", "working"):
            self._a.stop()
        self.setVisible(bool(text))
        self.update()

    def paintEvent(self, e) -> None:
        if not self._text:
            return
        p = theme.P()
        col = {"live": p.record, "working": p.accent, "done": p.success, "warn": p.warning, "idle": p.text3}[self._kind]
        g = QPainter(self)
        g.setRenderHint(QPainter.RenderHint.Antialiasing)
        c = QPointF(8, self.height() / 2)
        if self._kind in ("live", "working"):
            g.setPen(Qt.PenStyle.NoPen)
            g.setBrush(QBrush(qcolor(col, 0.25 * (1 - self._ph))))
            rr = 4 + 4 * self._ph
            g.drawEllipse(c, rr, rr)
        g.setPen(Qt.PenStyle.NoPen)
        g.setBrush(QBrush(qcolor(col)))
        g.drawEllipse(c, 4, 4)
        f = QFont(self.font()); f.setPixelSize(TYPE["footnote"]); f.setWeight(QFont.Weight.Medium)
        g.setFont(f)
        g.setPen(qcolor(p.text2))
        g.drawText(QRectF(22, 0, self.width() - 22, self.height()), Qt.AlignmentFlag.AlignVCenter, self._text)


# ----------------------------------------------------------------------------- busy indicator (logo mark, animated)
class LogoSpinner(QWidget):
    """The 5-bar MeetAssist mark, animated: bars bounce like an equalizer, then gather into a diamond silhouette, on a loop."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._n = 5
        self._phase = 0.0
        self.setFixedSize(46, 24)
        self._a = QVariantAnimation(self, duration=2400, startValue=0.0, endValue=1.0, loopCount=-1)
        self._a.valueChanged.connect(lambda v: (setattr(self, "_phase", v), self.update()))
        theme.init().changed.connect(self.update)

    def start(self) -> None:
        if self._a.state() != QVariantAnimation.State.Running:
            self._phase = 0.0; self._a.start()

    def stop(self) -> None:
        self._a.stop()

    def paintEvent(self, e) -> None:
        p = theme.P()
        g = QPainter(self)
        g.setRenderHint(QPainter.RenderHint.Antialiasing)
        n, bw, gap = self._n, 4.0, 3.4
        total = n * bw + (n - 1) * gap
        x0, cy = (self.width() - total) / 2, self.height() / 2
        ph = self._phase
        if ph < 0.45:                                                  # wave -> morph in -> diamond hold -> morph out -> wave
            blend = 0.0
        elif ph < 0.60:
            blend = (ph - 0.45) / 0.15
        elif ph < 0.80:
            blend = 1.0
        elif ph < 0.95:
            blend = 1.0 - (ph - 0.80) / 0.15
        else:
            blend = 0.0
        maxh = self.height() * 0.92
        mid = (n - 1) / 2
        for i in range(n):
            wave_h = maxh * (0.30 + 0.32 * (0.5 + 0.5 * math.sin(ph * 2 * math.pi * 1.6 + i * 1.05)))
            diamond_h = maxh * (0.22 + 0.78 * (1.0 - abs(i - mid) / mid))
            hh = max(bw, wave_h * (1 - blend) + diamond_h * blend)
            cx = x0 + i * (bw + gap) + bw / 2
            col = QColor(p.record if i == mid else p.accent)
            g.setPen(Qt.PenStyle.NoPen); g.setBrush(QBrush(col))
            g.drawRoundedRect(QRectF(cx - bw / 2, cy - hh / 2, bw, hh), bw / 2, bw / 2)


class JobIndicator(QWidget):
    """Shown above the tab content while a background LLM job runs: the animated mark + status text ("Thinking…", "Summarising…")."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        lay = QHBoxLayout(self); lay.setContentsMargins(0, SP["xs"], 0, SP["xs"]); lay.setSpacing(SP["sm"])
        self.spinner = LogoSpinner()
        self.lbl = label("", "footnote")
        lay.addWidget(self.spinner); lay.addWidget(self.lbl); lay.addStretch()
        self.hide()

    def start(self, text: str) -> None:
        self.lbl.setText(text)
        self.spinner.start()
        self.show()

    def set_text(self, text: str) -> None:
        self.lbl.setText(text)

    def stop(self) -> None:
        self.spinner.stop()
        self.hide()


class Banner(QFrame):
    """Inline notice (warning / error / info) with icon, text and optional close."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("banner")
        lay = QHBoxLayout(self); lay.setContentsMargins(SP["lg"], SP["md"], SP["md"], SP["md"]); lay.setSpacing(SP["md"])
        self.ic = QLabel(); self.ic.setFixedSize(theme.ICON["md"], theme.ICON["md"])
        self.tx = QLabel(); self.tx.setWordWrap(True)
        lay.addWidget(self.ic, 0, Qt.AlignmentFlag.AlignTop); lay.addWidget(self.tx, 1)
        self._kind = "warn"
        self.setAccessibleName("Notice")
        self.hide()
        theme.init().changed.connect(self._style)

    def show_message(self, text: str, kind: str = "warn") -> None:
        if not text:
            self.hide(); return
        self._kind = kind
        self.tx.setText(text)
        self._style()
        self.show()

    def _style(self) -> None:
        p = theme.P()
        bg, fg, ic = {"warn": (p.warning_tint, p.warning, "alert"), "error": (p.record_tint, p.record_hover if not p.dark else p.record, "alert"),
                      "info": (p.accent_tint, p.accent, "info")}[self._kind]
        self.setStyleSheet(f"QFrame#banner {{ background: {bg}; border-radius: {RADIUS['md']}px; }} QLabel {{ color: {p.text}; background: transparent; }}")
        self.ic.setPixmap(icons.pixmap(ic, fg, theme.ICON["md"]))


class Toast(QLabel):
    """Small dark pill that fades in/out at the bottom centre of its parent."""

    def __init__(self, parent: QWidget) -> None:
        super().__init__(parent)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.hide()
        self._fx = QGraphicsOpacityEffect(self)
        self.setGraphicsEffect(self._fx)
        self._anim = QPropertyAnimation(self._fx, b"opacity", self)
        self._timer = QTimer(self, singleShot=True, timeout=self._fade_out)
        self.setAccessibleName("Notification")

    def show_message(self, text: str, ms: int = 2600) -> None:
        p = theme.P()
        self.setStyleSheet(f"background: {p.text}; color: {p.bg}; border-radius: 17px; padding: 8px 18px; font-size: {TYPE['footnote']}px; font-weight: 600;")
        self.setText(text)
        self.adjustSize()
        par = self.parentWidget()
        self.move((par.width() - self.width()) // 2, par.height() - self.height() - 28)
        self.raise_(); self.show()
        self._anim.stop(); self._anim.setDuration(MOTION["base"]); self._anim.setStartValue(0.0); self._anim.setEndValue(1.0); self._anim.start()
        self._timer.start(ms)

    def _fade_out(self) -> None:
        self._anim.stop(); self._anim.setDuration(MOTION["slow"]); self._anim.setStartValue(self._fx.opacity()); self._anim.setEndValue(0.0)
        self._anim.finished.connect(self._done); self._anim.start()

    def _done(self) -> None:
        try:
            self._anim.finished.disconnect(self._done)
        except Exception:
            pass
        self.hide()


# ----------------------------------------------------------------------------- empty state
class EmptyState(QWidget):
    def __init__(self, icon_name: str, title: str, body: str, parent=None) -> None:
        super().__init__(parent)
        self._icon = icon_name
        lay = QVBoxLayout(self); lay.setAlignment(Qt.AlignmentFlag.AlignCenter); lay.setSpacing(SP["md"])
        self.badge = QLabel(); self.badge.setFixedSize(88, 88); self.badge.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.title = label(title, "hero"); self.title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.body = label(body, "body2", wrap=True); self.body.setAlignment(Qt.AlignmentFlag.AlignCenter); self.body.setMaximumWidth(440); self.body.setMinimumHeight(60)
        self.actions = QHBoxLayout(); self.actions.setAlignment(Qt.AlignmentFlag.AlignCenter); self.actions.setSpacing(SP["md"])
        lay.addWidget(self.badge, 0, Qt.AlignmentFlag.AlignCenter); lay.addSpacing(SP["sm"])
        lay.addWidget(self.title); lay.addWidget(self.body, 0, Qt.AlignmentFlag.AlignCenter); lay.addSpacing(SP["md"]); lay.addLayout(self.actions)
        theme.init().changed.connect(self._paint)
        self._paint()

    def set_text(self, title: str, body: str) -> None:
        self.title.setText(title)
        self.body.setText(body)
        self.body.setFixedWidth(440)
        self.body.setMinimumHeight(self.body.heightForWidth(440) + 4)       # wrapped labels need an explicit height

    def _paint(self) -> None:
        p = theme.P()
        self.badge.setStyleSheet(f"background: {p.accent_tint}; border-radius: 44px;")
        self.badge.setPixmap(icons.pixmap(self._icon, p.accent, 38, stroke=1.6))


# ----------------------------------------------------------------------------- meeting list rows
ROLE_PATH, ROLE_TITLE, ROLE_SUB, ROLE_STATE, ROLE_HEADER = (Qt.ItemDataRole.UserRole + i for i in range(5))


class MeetingDelegate(QStyledItemDelegate):
    ROW_H, HEAD_H = 58, 34

    def sizeHint(self, option, index) -> QSize:
        return QSize(option.rect.width(), self.HEAD_H if index.data(ROLE_HEADER) else self.ROW_H)

    def paint(self, g: QPainter, option, index) -> None:
        p = theme.P()
        g.save()
        g.setRenderHint(QPainter.RenderHint.Antialiasing)
        r = QRectF(option.rect)
        if index.data(ROLE_HEADER):
            f = QFont(option.font); f.setPixelSize(TYPE["footnote"]); f.setWeight(QFont.Weight.DemiBold)
            g.setFont(f); g.setPen(qcolor(p.text3))
            g.drawText(r.adjusted(18, 8, -8, 0), Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, index.data(ROLE_TITLE))
            g.restore(); return
        card = r.adjusted(8, 2, -8, -2)
        sel = bool(option.state & QStyle.StateFlag.State_Selected)
        hov = bool(option.state & QStyle.StateFlag.State_MouseOver)
        if sel or hov:
            g.setPen(Qt.PenStyle.NoPen)
            g.setBrush(QBrush(qcolor(p.accent_tint if sel else p.fill, 1.0)))
            g.drawRoundedRect(card, RADIUS["md"], RADIUS["md"])
        if option.state & QStyle.StateFlag.State_HasFocus and sel:
            g.setBrush(Qt.BrushStyle.NoBrush); g.setPen(QPen(qcolor(p.accent), 2))
            g.drawRoundedRect(card.adjusted(1, 1, -1, -1), RADIUS["md"], RADIUS["md"])
        state = index.data(ROLE_STATE) or "done"
        x0 = card.left() + 14
        tw = card.width() - 28
        if state in ("live", "busy"):
            col = p.record if state == "live" else p.accent
            g.setPen(Qt.PenStyle.NoPen); g.setBrush(QBrush(qcolor(col)))
            g.drawEllipse(QPointF(card.right() - 16, card.top() + 20), 4, 4)
            tw -= 18
        f = QFont(option.font); f.setPixelSize(TYPE["body"]); f.setWeight(QFont.Weight.DemiBold)
        g.setFont(f); g.setPen(qcolor(p.text))
        fm = QFontMetrics(f)
        g.drawText(QPointF(x0, card.top() + 23), fm.elidedText(index.data(ROLE_TITLE) or "", Qt.TextElideMode.ElideRight, int(tw)))
        f2 = QFont(option.font); f2.setPixelSize(TYPE["caption"] + 0.5)
        g.setFont(f2); g.setPen(qcolor(p.text3))
        sub = index.data(ROLE_SUB) or ""
        g.drawText(QPointF(x0, card.top() + 42), QFontMetrics(f2).elidedText(sub, Qt.TextElideMode.ElideRight, int(card.width() - 28)))
        g.restore()


# ----------------------------------------------------------------------------- switch (iOS-style toggle)
class Switch(QAbstractButton):
    def __init__(self, checked: bool = False, parent=None) -> None:
        super().__init__(parent)
        self.setCheckable(True)
        self.setChecked(checked)
        self.setFixedSize(44, 26)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self._x = 1.0 if checked else 0.0
        self._a = QVariantAnimation(self, duration=MOTION["base"], easingCurve=QEasingCurve.Type.OutCubic)
        self._a.valueChanged.connect(lambda v: (setattr(self, "_x", v), self.update()))
        self.toggled.connect(self._go)
        theme.init().changed.connect(self.update)

    def _go(self, on: bool) -> None:
        self._a.stop(); self._a.setStartValue(self._x); self._a.setEndValue(1.0 if on else 0.0); self._a.start()

    def paintEvent(self, e) -> None:
        p = theme.P()
        g = QPainter(self)
        g.setRenderHint(QPainter.RenderHint.Antialiasing)
        r = QRectF(self.rect()).adjusted(1, 1, -1, -1)
        off, on = QColor(p.fill_press if not p.dark else p.fill_hover), QColor(p.success if not p.dark else p.success)
        t = self._x
        col = QColor(int(off.red() + (on.red() - off.red()) * t), int(off.green() + (on.green() - off.green()) * t), int(off.blue() + (on.blue() - off.blue()) * t))
        g.setPen(Qt.PenStyle.NoPen); g.setBrush(QBrush(col))
        g.drawRoundedRect(r, r.height() / 2, r.height() / 2)
        d = r.height() - 4
        g.setBrush(QBrush(QColor("#FFFFFF")))
        g.drawEllipse(QRectF(r.left() + 2 + t * (r.width() - d - 4), r.top() + 2, d, d))
        if self.hasFocus():
            g.setBrush(Qt.BrushStyle.NoBrush); g.setPen(QPen(qcolor(p.accent), 2))
            g.drawRoundedRect(QRectF(self.rect()).adjusted(0, 0, 0, 0).adjusted(1, 1, -1, -1), 13, 13)

    def keyPressEvent(self, e) -> None:
        if e.key() in (Qt.Key.Key_Space, Qt.Key.Key_Return):
            self.toggle()
        else:
            super().keyPressEvent(e)
