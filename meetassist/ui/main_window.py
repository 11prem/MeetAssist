"""Main window - Apple-style: meetings sidebar, large reading area, floating recorder dock."""
from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import List, Optional

import numpy as np
from PySide6.QtCore import QEasingCurve, QEvent, QThread, QTimer, Qt, QUrl, QVariantAnimation, Signal
from PySide6.QtGui import QAction, QActionGroup, QDesktopServices, QGuiApplication, QKeySequence, QPixmap, QShortcut, QFont
from PySide6.QtWidgets import (QApplication, QFileDialog, QFrame, QHBoxLayout, QInputDialog, QLabel, QLineEdit, QListWidget,
                               QListWidgetItem, QMainWindow, QMenu, QMessageBox, QPlainTextEdit, QProgressBar,
                               QStackedWidget, QVBoxLayout, QWidget)

from .. import export, paths, store, voiceid
from ..controller import Controller
from ..llm import LLMClient
from ..segmenter import RATE
from ..settings import Settings
from ..transcriber import available_models
from . import theme
from .icons import icon as svg_icon, pixmap
from .name_speaker_dialog import NameSpeakerDialog
from .render import ReadingView, chat_html, markdown_html, placeholder_html, summary_css, transcript_html
from .settings_dialog import SettingsDialog, _RecordVoice
from .theme import SP, TYPE
from .widgets import (ROLE_HEADER, ROLE_PATH, ROLE_STATE, ROLE_SUB, ROLE_TITLE, ActionBar, Banner, Composer, EmptyState,
                      IconButton, JobIndicator, LabelButton, MeetingDelegate, MeterRow, PopButton, RecordButton,
                      Segmented, SideButton, SourceStatusLabel, StatusPill, TitleBar, Toast, WaveStrip, label, shadow)

RESIZE_MARGIN = 6

DISPLAY_NAMES = {"english": "English", "roman": "Tanglish", "tamil": "Tamil"}


def fmt_dur(sec: float) -> str:
    s = int(sec)
    return f"{s // 3600}:{s % 3600 // 60:02d}:{s % 60:02d}" if s >= 3600 else f"{s // 60}:{s % 60:02d}"


def fmt_clock(sec: float) -> str:
    s = int(sec)
    return f"{s // 3600:02d}:{s % 3600 // 60:02d}:{s % 60:02d}"


def level01(rms: Optional[float]) -> float:
    if not rms or rms <= 0:
        return 0.0
    return max(0.0, min(1.0, (20 * math.log10(rms) + 55) / 55))


def day_group(epoch: float) -> str:
    now = time.localtime()
    d0 = time.mktime((now.tm_year, now.tm_mon, now.tm_mday, 0, 0, 0, 0, 0, -1))
    if epoch >= d0:
        return "Today"
    if epoch >= d0 - 86400:
        return "Yesterday"
    if epoch >= d0 - 6 * 86400:
        return "Previous 7 days"
    return "Earlier"


class Job(QThread):
    """Run a blocking callable (LLM / file work) off the GUI thread."""
    done = Signal(object)
    failed = Signal(str)
    progress = Signal(int, int)

    def __init__(self, fn, parent=None) -> None:
        super().__init__(parent)
        self.fn = fn

    def run(self) -> None:
        try:
            self.done.emit(self.fn(lambda a, b: self.progress.emit(a, b)))
        except Exception as e:                                 # shown to the user, never crashes the app
            self.failed.emit(str(e))


class MainWindow(QMainWindow):
    def __init__(self, settings: Settings) -> None:
        super().__init__()
        self.settings = settings
        self.ctrl = Controller(settings)
        self.current: Optional[Path] = None
        self._row_count = -1
        self._jobs: List[Job] = []
        self._job_text = ""
        self._finalized: set = set()
        self._list_sig = None
        self._search = ""
        self.setWindowTitle("MeetAssist")
        self.setWindowFlag(Qt.WindowType.FramelessWindowHint)   # custom title bar below owns move/resize/min/max/close
        self.setMinimumSize(900, 540)
        self._fit_to_screen()
        self.tm = theme.init()
        self._meter_t = QTimer(self, interval=80, timeout=self._meter_tick)
        self.tm.set_mode(settings.appearance)
        self._build()
        self.tm.changed.connect(self._apply_theme)
        self._apply_theme()
        self._shortcuts()
        self.ctrl.recover()
        self.reload_meetings()
        QTimer(self, interval=600, timeout=self.refresh).start()
        QTimer(self, interval=1000, timeout=self.ctrl.tick).start()
        QTimer(self, interval=4000, timeout=self.reload_meetings).start()
        QTimer.singleShot(300, self._first_run_check)
        self.refresh()
        self.setFocus(Qt.FocusReason.OtherFocusReason)       # no focus ring on first paint; Tab reaches every control

    def _fit_to_screen(self) -> None:
        """Open at 1240x800 but never larger than ~92 % of the usable screen (small laptops: 1366x768), centred."""
        scr = self.screen() or QGuiApplication.primaryScreen()
        av = scr.availableGeometry()
        w, h = min(1240, int(av.width() * 0.92)), min(800, int(av.height() * 0.92))
        self.resize(max(w, 900), max(h, 540))
        self.move(av.left() + (av.width() - self.width()) // 2, av.top() + (av.height() - self.height()) // 2)

    # ================================================================== frameless window: edge resize
    def _edge_at(self, pos) -> tuple:
        """left/right/bottom only: the body's own top margin is an internal gap below the title
        bar, not a window edge, so top-edge resize isn't offered (drag a corner instead)."""
        m = RESIZE_MARGIN
        return pos.x() <= m, pos.x() >= self._body.width() - m, pos.y() >= self._body.height() - m

    def eventFilter(self, obj, e) -> bool:
        if obj is self._body:
            if e.type() == QEvent.Type.MouseMove and not e.buttons():
                left, right, bottom = self._edge_at(e.position().toPoint())
                cur = (Qt.CursorShape.SizeBDiagCursor if (left and bottom) else
                       Qt.CursorShape.SizeFDiagCursor if (right and bottom) else
                       Qt.CursorShape.SizeHorCursor if (left or right) else
                       Qt.CursorShape.SizeVerCursor if bottom else Qt.CursorShape.ArrowCursor)
                self._body.setCursor(cur)
            elif e.type() == QEvent.Type.MouseButtonPress and e.button() == Qt.MouseButton.LeftButton:
                left, right, bottom = self._edge_at(e.position().toPoint())
                if left or right or bottom:
                    edges = Qt.Edge(0)
                    if left: edges |= Qt.Edge.LeftEdge
                    if right: edges |= Qt.Edge.RightEdge
                    if bottom: edges |= Qt.Edge.BottomEdge
                    wh = self.windowHandle()
                    if wh is not None:
                        wh.startSystemResize(edges); return True
        return super().eventFilter(obj, e)

    # ================================================================== construction
    def _build(self) -> None:
        central = QWidget(); central.setObjectName("mainbg"); self.setCentralWidget(central)
        outer = QVBoxLayout(central); outer.setContentsMargins(0, 0, 0, 0); outer.setSpacing(0)
        self.titlebar = TitleBar(); outer.addWidget(self.titlebar)
        self._body = QWidget(); self._body.setMouseTracking(True); self._body.installEventFilter(self)
        outer.addWidget(self._body, 1)
        root = QHBoxLayout(self._body); root.setContentsMargins(SP["md"], SP["md"], SP["md"], SP["md"]); root.setSpacing(SP["md"])

        # ------------------------------------------------ sidebar
        side = QWidget(); side.setObjectName("sidebar"); side.setFixedWidth(300)
        sv = QVBoxLayout(side); sv.setContentsMargins(SP["md"], SP["lg"], SP["md"], SP["md"]); sv.setSpacing(SP["md"])
        brand = QHBoxLayout(); brand.setContentsMargins(SP["sm"], 0, 0, 0); brand.setSpacing(SP["md"])
        self.logo = QLabel(); self.logo.setFixedSize(28, 28); self.logo.setAccessibleName("MeetAssist")
        brand.addWidget(self.logo); brand.addWidget(label("MeetAssist", "headline")); brand.addStretch()
        sv.addLayout(brand)
        self.search = QLineEdit(); self.search.setPlaceholderText("Search meetings"); self.search.setClearButtonEnabled(True)
        self.search.setAccessibleName("Search meetings"); self.search.textChanged.connect(self._on_search)
        sv.addWidget(self.search)
        self.listw = QListWidget(); self.list_delegate = MeetingDelegate(self.listw)
        self.listw.setItemDelegate(self.list_delegate); self.listw.setMouseTracking(True)
        self._row_hover_anim = QVariantAnimation(self, duration=160, easingCurve=QEasingCurve.Type.OutBack)
        self._row_hover_anim.valueChanged.connect(self._on_row_hover_scale)
        self.listw.entered.connect(self._on_row_entered); self.listw.viewportEntered.connect(self._on_row_left)
        self.listw.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.listw.setVerticalScrollMode(QListWidget.ScrollMode.ScrollPerPixel); self.listw.setAccessibleName("Meetings")
        self.listw.currentItemChanged.connect(self._on_select)
        self.listw.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu); self.listw.customContextMenuRequested.connect(self._list_menu)
        sv.addWidget(self.listw, 1)
        sep = QFrame(); sep.setObjectName("hsep"); sv.addWidget(sep)
        self.b_import = SideButton("upload", "Import audio…"); self.b_import.clicked.connect(self.import_audio)
        self.b_settings = SideButton("sliders", "Settings"); self.b_settings.clicked.connect(self.open_settings)
        sv.addWidget(self.b_import); sv.addWidget(self.b_settings)
        root.addWidget(side)
        vs = QFrame(); vs.setObjectName("vsep"); root.addWidget(vs)

        # ------------------------------------------------ content
        content = QWidget(); content.setObjectName("content"); root.addWidget(content, 1)
        cv = QVBoxLayout(content); cv.setContentsMargins(0, 0, 0, 0); cv.setSpacing(SP["md"])
        self.pages = QStackedWidget(); cv.addWidget(self.pages, 1)

        # page 0: empty state
        self.empty = EmptyState("waveform", "Record your first meeting",
                                "MeetAssist captures your microphone and everyone on the call, then writes the transcript on this computer — "
                                "Tamil, English and Tanglish. Nothing leaves your laptop.")
        self.empty_import = PopButton("Import an audio file…"); self.empty_import.clicked.connect(self.import_audio)
        self.empty.actions.addWidget(self.empty_import)
        self.pages.addWidget(self.empty)

        # page 1: meeting
        page = QWidget(); page.setObjectName("page"); pv = QVBoxLayout(page); pv.setContentsMargins(SP["xxl"], SP["xl"], SP["xxl"], 0); pv.setSpacing(SP["md"])
        t1 = QHBoxLayout(); t1.setSpacing(SP["sm"])
        self.head = label("", "title"); self.head.setAccessibleName("Meeting title")
        self.b_rename = IconButton("pencil", "Rename meeting", size=theme.ICON["md"]); self.b_rename.clicked.connect(lambda: self.current and self._rename(self.current))
        t1.addWidget(self.head); t1.addWidget(self.b_rename); t1.addStretch(); pv.addLayout(t1)
        t2 = QHBoxLayout(); t2.setSpacing(SP["md"])
        self.pill = StatusPill(); self.detail = label("", "footnote")
        t2.addWidget(self.pill); t2.addWidget(self.detail); t2.addStretch(); pv.addLayout(t2)
        self.prog = QProgressBar(); self.prog.setTextVisible(False); self.prog.hide(); pv.addWidget(self.prog)
        self.banner = Banner(); pv.addWidget(self.banner)

        tb = QHBoxLayout(); tb.setSpacing(SP["xs"])
        self.seg = Segmented(["Transcript", "Cleaned", "Summary", "Ask"]); self.seg.currentChanged.connect(self._on_tab)
        tb.addWidget(self.seg); tb.addStretch()
        self.b_view = LabelButton("languages", DISPLAY_NAMES[self.settings.tamil_display], "How Tamil speech is shown")
        vmenu = QMenu(self); grp = QActionGroup(self); grp.setExclusive(True)
        for key, text in (("english", "Tamil as English translation"), ("roman", "Tamil in English letters (Tanglish)"), ("tamil", "Tamil in Tamil script")):
            a = QAction(text, self, checkable=True); a.setData(key); a.setChecked(key == self.settings.tamil_display)
            a.triggered.connect(lambda _=False, k=key: self._set_display(k)); grp.addAction(a); vmenu.addAction(a)
        self.b_view.setMenu(vmenu)
        self.b_copy = IconButton("copy", "Copy transcript"); self.b_copy.clicked.connect(self.copy_text)
        self.b_save = IconButton("download", "Save as…  (Ctrl+S)"); self.b_save.clicked.connect(self.save_as)
        self.b_folder = IconButton("folder", "Open meeting folder"); self.b_folder.clicked.connect(self.open_folder)
        self.b_more = IconButton("more", "More"); self.more_menu = QMenu(self); self.b_more.clicked.connect(self._open_more)
        for w in (self.b_view, self.b_copy, self.b_save, self.b_folder, self.b_more):
            tb.addWidget(w)
        pv.addLayout(tb)
        sep2 = QFrame(); sep2.setObjectName("hsep"); pv.addWidget(sep2)

        self.job_indicator = JobIndicator(); pv.addWidget(self.job_indicator)
        self.views = QStackedWidget(); pv.addWidget(self.views, 1)
        raw_page = QWidget(); raw_v = QVBoxLayout(raw_page); raw_v.setContentsMargins(0, 0, 0, 0); raw_v.setSpacing(SP["sm"])
        self.view_raw = ReadingView(); raw_v.addWidget(self.view_raw, 1)
        self.b_clean = PopButton("Clean up"); self.b_clean.setProperty("variant", "primary")
        self.b_clean.setIcon(svg_icon("sparkles", theme.P().on_accent)); self.b_clean.clicked.connect(lambda: self.run_clean())
        raw_v.addWidget(ActionBar(self.b_clean))
        self.views.addWidget(raw_page)

        clean_page = QWidget(); clean_v = QVBoxLayout(clean_page); clean_v.setContentsMargins(0, 0, 0, 0); clean_v.setSpacing(SP["sm"])
        self.view_clean = QPlainTextEdit()
        self.view_clean.setPlaceholderText("Turn on “Clean up automatically” in Settings to have this fill in on its own, or "
                                           "click Clean up on the Transcript tab to run it now. The corrected text will appear "
                                           "here - fix anything the AI got wrong (misheard names, jargon, etc.) directly in this box.")
        self.view_clean.textChanged.connect(self._on_clean_edited)
        clean_v.addWidget(self.view_clean, 1)
        self.b_summarize = PopButton("Summarize"); self.b_summarize.setProperty("variant", "primary")
        self.b_summarize.setIcon(svg_icon("sparkles", theme.P().on_accent)); self.b_summarize.clicked.connect(lambda: self.run_summary())
        clean_v.addWidget(ActionBar(self.b_summarize))
        self.views.addWidget(clean_page)

        sum_page = QWidget(); sum_v = QVBoxLayout(sum_page); sum_v.setContentsMargins(0, 0, 0, 0); sum_v.setSpacing(SP["sm"])
        self.sum_views = QStackedWidget()
        self.view_sum_preview = ReadingView(); self.sum_views.addWidget(self.view_sum_preview)
        self.view_sum = QPlainTextEdit()
        self.view_sum.setPlaceholderText("No summary yet. Review the Cleaned tab, then click Summarize.")
        self.view_sum.textChanged.connect(self._on_summary_edited)
        self.sum_views.addWidget(self.view_sum)
        sum_v.addWidget(self.sum_views, 1)
        self.b_sum_edit = PopButton("Edit"); self.b_sum_edit.clicked.connect(self._toggle_summary_edit)
        sum_v.addWidget(ActionBar(self.b_sum_edit))
        self.views.addWidget(sum_page)

        ask_page = QWidget(); ask_v = QVBoxLayout(ask_page); ask_v.setContentsMargins(0, 0, 0, 0); ask_v.setSpacing(SP["sm"])
        self.chat_log = ReadingView(); ask_v.addWidget(self.chat_log, 1)
        self.composer = Composer("Ask a question about this meeting…")
        self.composer.input.setAccessibleName("Ask about this meeting")
        self.composer.send.connect(self._ask_question)
        ask_v.addWidget(self.composer)
        self.views.addWidget(ask_page)

        self._chat_history: List[dict] = []
        self._clean_save_t = QTimer(self, interval=700, singleShot=True); self._clean_save_t.timeout.connect(self._save_clean)
        self._summary_save_t = QTimer(self, interval=700, singleShot=True); self._summary_save_t.timeout.connect(self._save_summary)
        self._loading_side = False
        self.pages.addWidget(page)

        # ------------------------------------------------ recorder dock (always visible, never overlaps content)
        dock_wrap = QWidget(); dock_wrap.setObjectName("content")
        dw = QVBoxLayout(dock_wrap); dw.setContentsMargins(SP["xxl"], SP["md"], SP["xxl"], SP["xl"])
        self.dock = QFrame(); self.dock.setObjectName("dock"); self.dock.setMinimumHeight(88)
        dl = QHBoxLayout(self.dock); dl.setContentsMargins(SP["lg"], SP["md"], SP["xl"], SP["md"]); dl.setSpacing(SP["lg"])
        self.rec = RecordButton(); self.rec.clicked.connect(self.toggle_record); dl.addWidget(self.rec)
        mid = QVBoxLayout(); mid.setSpacing(SP["xs"]); mid.setAlignment(Qt.AlignmentFlag.AlignVCenter)
        self.dock_stack = QStackedWidget(); self.dock_stack.setFixedHeight(28); self.dock_stack.setFixedWidth(230)
        self.title_edit = QLineEdit(); self.title_edit.setProperty("flat", True); self.title_edit.setPlaceholderText("Name this meeting (optional)")
        self.title_edit.setAccessibleName("Meeting title")
        self.timer_lbl = QLabel("00:00:00"); f = theme.tabular(QFont(self.timer_lbl.font())); f.setPixelSize(TYPE["timer"]); f.setWeight(QFont.Weight.DemiBold)
        self.timer_lbl.setFont(f); self.timer_lbl.setAccessibleName("Recording time")
        self.dock_stack.addWidget(self.title_edit); self.dock_stack.addWidget(self.timer_lbl)
        self.dock_sub = label("", "footnote")
        self.sources_label = SourceStatusLabel(self.settings.record_mic, self.settings.record_system_audio)
        self.sub_stack = QStackedWidget()
        self.sub_stack.addWidget(self.sources_label); self.sub_stack.addWidget(self.dock_sub)
        mid.addWidget(self.dock_stack); mid.addWidget(self.sub_stack); dl.addLayout(mid)
        self.wave = WaveStrip(); dl.addWidget(self.wave, 1)
        meters = QVBoxLayout(); meters.setSpacing(SP["sm"]); meters.setAlignment(Qt.AlignmentFlag.AlignVCenter)
        self.m_mic = MeterRow("mic", "you", "You (microphone)"); self.m_loop = MeterRow("speaker", "remote", "Remote (system audio)")
        self.m_mic.set_muted(not self.settings.record_mic); self.m_loop.set_muted(not self.settings.record_system_audio)
        self.m_mic.clicked.connect(lambda: self._toggle_source("mic")); self.m_loop.clicked.connect(lambda: self._toggle_source("system"))
        meters.addWidget(self.m_mic); meters.addWidget(self.m_loop)
        mw = QWidget(); mw.setLayout(meters); mw.setFixedWidth(170); dl.addWidget(mw)
        dw.addWidget(self.dock)
        cv.addWidget(dock_wrap)
        shadow(self.dock)

        self.toast = Toast(content)

    def _shortcuts(self) -> None:
        for seq, fn in (("Ctrl+R", self.toggle_record), ("Ctrl+O", self.import_audio), ("Ctrl+,", self.open_settings),
                        ("Ctrl+F", lambda: self.search.setFocus()), ("Ctrl+S", self.save_as), ("Ctrl+Shift+C", self.copy_text)):
            QShortcut(QKeySequence(seq), self, activated=fn)

    # ================================================================== theme
    def _apply_theme(self) -> None:
        p = self.tm.p
        self.setStyleSheet(theme.build_qss(p))
        theme.style_titlebar(self, p)
        from .icon import make_image
        dpr = self.devicePixelRatioF()
        pm = QPixmap.fromImage(make_image(int(28 * dpr))); pm.setDevicePixelRatio(dpr)      # rendered natively: always crisp
        self.logo.setPixmap(pm)
        for v in (self.view_raw, self.chat_log, self.view_sum_preview):
            v.document().setDefaultStyleSheet(summary_css(p))
        shadow(self.dock)
        self._row_count = -1
        self._load_side_files()
        self.refresh()

    def showEvent(self, e) -> None:
        super().showEvent(e)
        theme.style_titlebar(self, self.tm.p)

    def _toast(self, msg: str) -> None:
        self.toast.show_message(msg)

    # ================================================================== first run / settings
    def _models_ok(self) -> bool:
        need_indic = self.settings.language in ("hybrid", "ta") and not (paths.MODELS_DIR / "indic-asr" / "assets" / "encoder.onnx").exists()
        return bool(available_models(paths.MODELS_DIR)) and not need_indic

    def _first_run_check(self) -> None:
        if not self._models_ok():
            QMessageBox.critical(self, "Speech model missing",
                                 f"No speech-recognition model was found in:\n{paths.MODELS_DIR}\n\nCopy the 'models' folder next to the application.")
            return
        self._maybe_offer_voice_enrollment()

    def _maybe_offer_voice_enrollment(self) -> None:
        """One-time, on first run: offer to learn the user's voice so MeetAssist never has to ask
        who they are, even in an in-person meeting where everyone shares one microphone."""
        if self.settings.voice_enroll_asked or any(p["id"] == voiceid.ME_ID for p in voiceid.load_profiles()):
            return
        self.settings.voice_enroll_asked = True
        self.settings.save()
        if QMessageBox.question(self, "Recognise your voice",
                                "MeetAssist can learn your voice once so it never has to ask who you are later - "
                                "even in an in-person meeting where everyone is heard through the same microphone.\n\n"
                                "Record a 10-second sample now? You can also do this later in Settings → Voices.",
                                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No) == QMessageBox.StandardButton.Yes:
            self._toast("Recording your voice for 10 seconds…")
            self._enroll = _RecordVoice(self.ctrl)
            self._enroll.done.connect(lambda ok: self._toast("Voice saved" if ok else "Could not record your voice"))
            self._enroll.start()

    def open_settings(self) -> None:
        dlg = SettingsDialog(self.settings, self.ctrl, self)
        if dlg.exec():
            self.settings.save(); self.ctrl.settings = self.settings
            self.tm.set_mode(self.settings.appearance)
            self._toast("Settings saved")

    def _set_display(self, key: str) -> None:
        self.settings.tamil_display = key; self.settings.save()
        self.b_view.setText(DISPLAY_NAMES[key])
        self._row_count = -1; self._load_side_files(); self.refresh()

    # ================================================================== recording
    def toggle_record(self) -> None:
        if self.ctrl.active:
            self.rec.set_busy(True)
            QApplication.processEvents()
            d = self.ctrl.stop_live()
            self.rec.set_busy(False)
            self.select_meeting(d)
            return
        if not self._models_ok():
            self._first_run_check(); return
        if not (self.settings.record_mic or self.settings.record_system_audio):
            QMessageBox.warning(self, "Nothing to record", "Turn on the microphone and/or system audio in Settings."); return
        if not self.settings.ack_notice:
            box = QMessageBox(QMessageBox.Icon.Information, "Before you record",
                              "MeetAssist records your microphone and the sound playing on this computer.\n\n"
                              "Make sure everyone in the meeting knows it is being recorded, and that recording is allowed "
                              "for this meeting under your company's policy (client meetings may need Compliance approval).\n\n"
                              "Audio and transcripts stay on this computer. Only text you choose to send to the company LLM leaves it.",
                              QMessageBox.StandardButton.Ok | QMessageBox.StandardButton.Cancel, self)
            if box.exec() != QMessageBox.StandardButton.Ok:
                return
            self.settings.ack_notice = True; self.settings.save()
        d = self.ctrl.start_live(self.title_edit.text().strip())
        self.title_edit.clear(); self._row_count = -1
        self.reload_meetings(); self.select_meeting(d)

    def _meter_tick(self) -> None:
        a = self.ctrl.active
        if not a:
            return
        c = store.read_json(a / "raw" / "status.json") or {}
        m, l = level01(c.get("mic_rms")), level01(c.get("loop_rms"))
        self.m_mic.set_value(m); self.m_loop.set_value(l); self.wave.push(m, l)

    def _toggle_source(self, which: str) -> None:
        if self.ctrl.active:                               # takes effect on the next recording, not mid-capture
            return
        mic, sysa = self.settings.record_mic, self.settings.record_system_audio
        mic, sysa = (not mic, sysa) if which == "mic" else (mic, not sysa)
        if not (mic or sysa):
            self._toast("Keep at least one audio source on"); return
        self.settings.record_mic, self.settings.record_system_audio = mic, sysa
        self.settings.save()
        self.m_mic.set_muted(not mic); self.m_loop.set_muted(not sysa)
        self.sources_label.set_sources(mic, sysa)

    # ================================================================== meeting list
    def _on_search(self, text: str) -> None:
        self._search = text.strip().lower(); self._list_sig = None; self.reload_meetings()

    def _matches(self, m) -> bool:
        q = self._search
        if not q:
            return True
        if q in m.title.lower() or q in m.meta.get("started_at", "").lower():
            return True
        j = m.path / "journal.jsonl"
        try:
            return j.exists() and q in j.read_text(encoding="utf-8", errors="ignore").lower()
        except OSError:
            return False

    def reload_meetings(self) -> None:
        items = [m for m in store.list_meetings() if self._matches(m)]
        allm = store.list_meetings()
        rows = []
        for m in items:
            w = store.read_json(m.path / "worker.json") or {}
            live = m.meta.get("ended_at") is None
            state = "live" if live else ("busy" if w.get("state") in ("loading", "listening", "draining") else "done")
            dur = m.meta.get("duration") or 0
            ep = m.meta.get("epoch", 0)
            when = time.strftime("%H:%M" if day_group(ep) in ("Today", "Yesterday") else "%a %d %b  ·  %H:%M", time.localtime(ep))
            sub = ("Recording now" if live else f"{when}  ·  {fmt_dur(dur)}" + ("  ·  Transcribing" if state == "busy" else ""))
            rows.append((m, state, sub))
        sig = (self._search, self.current, [(m.id, m.title, st, sb) for m, st, sb in rows])
        if sig == self._list_sig:
            return
        self._list_sig = sig
        self.listw.blockSignals(True)
        self.listw.clear()
        last = None
        for m, st, sb in rows:
            g = day_group(m.meta.get("epoch", 0))
            if g != last:
                h = QListWidgetItem(); h.setData(ROLE_HEADER, True); h.setData(ROLE_TITLE, g); h.setFlags(Qt.ItemFlag.NoItemFlags); self.listw.addItem(h); last = g
            it = QListWidgetItem(); it.setData(ROLE_PATH, str(m.path)); it.setData(ROLE_TITLE, m.title); it.setData(ROLE_SUB, sb); it.setData(ROLE_STATE, st)
            it.setToolTip(m.title); self.listw.addItem(it)
            if self.current and m.path == self.current:
                self.listw.setCurrentItem(it)
        self.listw.blockSignals(False)
        self.empty.set_text("Record your first meeting" if not allm else "Select a meeting",
                            "MeetAssist captures your microphone and everyone on the call, then writes the transcript on this computer — Tamil, "
                            "English and Tanglish. Nothing leaves your laptop." if not allm else
                            "Pick a recording from the list, or press the red button to start a new one.")
        if self.current is None and rows and not self._search and not getattr(self, "_auto_done", False):
            self._auto_done = True                              # open the most recent meeting on launch
            self.current = rows[0][0].path
            for i in range(self.listw.count()):
                if self.listw.item(i).data(ROLE_PATH) == str(self.current):
                    self.listw.blockSignals(True); self.listw.setCurrentRow(i); self.listw.blockSignals(False)

    def _flush_edits(self) -> None:
        """Save a pending debounced edit immediately, before self.current changes to a different meeting."""
        for t, save in ((self._clean_save_t, self._save_clean), (self._summary_save_t, self._save_summary)):
            if t.isActive():
                t.stop(); save()

    def select_meeting(self, d: Path) -> None:
        self._flush_edits()
        self.current = d; self._row_count = -1; self._list_sig = None
        self.reload_meetings(); self._load_side_files(); self.refresh()

    def _on_select(self, item: Optional[QListWidgetItem], _prev=None) -> None:
        if item and item.data(ROLE_PATH):
            self._flush_edits()
            self.current = Path(item.data(ROLE_PATH)); self._row_count = -1
            self._load_side_files(); self.refresh()

    def _on_row_entered(self, index) -> None:
        path = index.data(ROLE_PATH)
        if not path or path == self.list_delegate.hover_path:
            return
        self.list_delegate.hover_path = path
        self._row_hover_anim.stop(); self._row_hover_anim.setStartValue(self.list_delegate.hover_scale)
        self._row_hover_anim.setEndValue(1.05); self._row_hover_anim.start()

    def _on_row_left(self) -> None:
        if self.list_delegate.hover_path is None:
            return
        self.list_delegate.hover_path = None
        self._row_hover_anim.stop(); self._row_hover_anim.setStartValue(self.list_delegate.hover_scale)
        self._row_hover_anim.setEndValue(1.0); self._row_hover_anim.start()

    def _on_row_hover_scale(self, v) -> None:
        self.list_delegate.hover_scale = v
        self.listw.viewport().update()

    def _list_menu(self, pos) -> None:
        it = self.listw.itemAt(pos)
        if not it or not it.data(ROLE_PATH):
            return
        d = Path(it.data(ROLE_PATH)); menu = QMenu(self)
        menu.addAction("Rename…", lambda: self._rename(d))
        menu.addAction("Open folder", lambda: QDesktopServices.openUrl(QUrl.fromLocalFile(str(d))))
        menu.addSeparator()
        menu.addAction("Delete meeting and audio…", lambda: self._delete(d))
        menu.exec(self.listw.mapToGlobal(pos))

    def _rename(self, d: Path) -> None:
        t, ok = QInputDialog.getText(self, "Rename meeting", "Title:", text=store.read_meta(d).get("title", ""))
        if ok and t.strip():
            store.update_meta(d, title=t.strip()); self._list_sig = None; self.reload_meetings(); self.refresh()

    def _delete(self, d: Path) -> None:
        if self.ctrl.active == d:
            QMessageBox.information(self, "Recording", "Stop the recording first."); return
        if QMessageBox.question(self, "Delete meeting", "Permanently delete this meeting, its transcript AND raw audio?") == QMessageBox.StandardButton.Yes:
            store.delete_meeting(d)
            if self.current == d:
                self.current = None
            self._list_sig = None; self.reload_meetings(); self.refresh()

    def _open_more(self) -> None:
        m = self.more_menu; m.clear()
        d = self.current
        sub = m.addMenu("Re-transcribe")
        for name in available_models(paths.MODELS_DIR):
            a = sub.addAction(f"Re-run English recognition with “{name}”"); a.triggered.connect(lambda _=False, n=name: self.retranscribe(n))
        sub.setEnabled(bool(d) and not self.ctrl.active == d and bool(store.audio_seconds(d)) if d else False)
        m.addSeparator()
        m.addAction("Rename…", lambda: d and self._rename(d)); m.addAction("Delete meeting and audio…", lambda: d and self._delete(d))
        m.exec(self.b_more.mapToGlobal(self.b_more.rect().bottomLeft()))

    # ================================================================== periodic refresh
    def _set_recording_ui(self, on: bool) -> None:
        self.rec.set_recording(on)
        self.dock_stack.setCurrentIndex(1 if on else 0)
        self.sub_stack.setCurrentIndex(1 if on else 0)
        self.wave.set_active(on)
        if on and not self._meter_t.isActive():
            self._meter_t.start()
        elif not on:
            self._meter_t.stop(); self.m_mic.set_value(0); self.m_loop.set_value(0)

    def refresh(self) -> None:
        if getattr(self, "_demo", False):                   # design-screenshot hook (tools/shot2.py)
            return
        a = self.ctrl.active
        self._set_recording_ui(bool(a))
        warn, wkind = "", "warn"
        if a:
            st = self.ctrl.poll(a); c = st["capture"]
            el = time.time() - store.read_meta(a).get("epoch", time.time())
            self.timer_lbl.setText(fmt_clock(el))
            self.dock_sub.setText("Recording  ·  " + store.read_meta(a).get("title", ""))
            warn = c.get("warning") or c.get("error") or ""
            if self.ctrl.capture_failed:
                warn, wkind = "Recording is NOT working: " + (c.get("error") or "audio device error") + ". Stop, fix the audio device, then start again.", "error"
            if not warn and self.settings.record_system_audio and c.get("loop_age") is None and el > 20:
                warn = "No system audio heard yet. Windows only provides it while something is playing through your speakers or headset."
            if not warn and self.ctrl.respawns:
                warn = f"Audio device changed. Recording continued ({self.ctrl.respawns}×)."
        else:
            self.sources_label.set_sources(self.settings.record_mic, self.settings.record_system_audio)
        while self.ctrl.refine_done:
            r = self.ctrl.refine_done.pop(0)
            self._toast("Improved transcript ready")
            if r == self.current:
                self._row_count = -1; self._load_side_files()
            self._postprocess(r)
        while self.ctrl.voiceid_done:
            r, unnamed = self.ctrl.voiceid_done.pop(0)
            if unnamed:
                self._prompt_speaker_names(r, unnamed)
            if r == self.current:
                self._row_count = -1
        d = self.current
        self.pages.setCurrentIndex(1 if d else 0)
        if not d:
            return
        st = self.ctrl.poll(d); w, rows = st["worker"], st["rows"]; meta = store.read_meta(d)
        self.head.setText(meta.get("title", d.name))
        state, live = w.get("state", ""), st["recording"]
        kind, text, detail = "idle", "", ""
        if live:
            behind = w.get("pending_s")
            kind, text = "live", "Recording"
            detail = "Transcribing live" + (f"  ·  {behind:.0f}s of speech queued" if behind else "")
            if behind and behind > 90:
                warn, wkind = "Transcription is falling behind. It will catch up after the meeting; “English only” or “Tamil only” mode keeps up on slower laptops.", "warn"
        elif state in ("loading", "listening", "draining"):
            ps = w.get("pending_s") or 0
            kind, text = "working", ("Loading models" if state == "loading" else "Transcribing")
            detail = "" if state == "loading" else f"{ps:.0f}s of speech left"
            if self.ctrl.import_progress is not None:
                detail = f"Reading audio {self.ctrl.import_progress * 100:.0f}%" + (f"  ·  {detail}" if detail else "")
        elif state == "done":
            kind, text = "done", "Transcript ready"
            detail = f"{fmt_dur(st['audio_s'])} of audio  ·  {w.get('rtf', 0):.2f}× real-time cost"
            if meta.get("refine_pending"):
                rs = self.ctrl.refine_state(d) or {}
                tot = sum(max(r["t_end"] - r["t_start"], 0) for r in rows) or 1
                kind, text = "working", "Improving accuracy"
                detail = f"{min(99, int(100 * (rs.get('speech_done_s') or 0) / tot))}%  ·  you can keep using the draft"
            if d not in self._finalized:
                self._finalized.add(d); self._on_done(d)
        elif state == "error":
            kind, text = "warn", "Transcription problem"; warn, wkind = w.get("error", "worker error"), "error"
        elif meta.get("ended_at") is None or not rows:
            kind, text = "working", "Starting up"
        self.pill.set(kind, text); self.detail.setText(detail)
        self.banner.show_message(warn, wkind)
        busy = state in ("loading", "listening", "draining") and not live
        has_prog = busy and self.ctrl.import_progress is not None
        self.prog.setVisible(has_prog)
        if has_prog:
            self.prog.setRange(0, 100); self.prog.setValue(int(self.ctrl.import_progress * 100))
        n = len(rows)
        if n != self._row_count:
            sb = self.view_raw.verticalScrollBar()
            stick = sb.value() >= sb.maximum() - 24
            self._render(self.view_raw, rows, "raw", None)
            if stick or self._row_count < 0:
                sb.setValue(sb.maximum())
            self._row_count = n
        can_ai = bool(rows) and not live
        for b in (self.b_copy, self.b_save):
            b.setEnabled(bool(rows))
        has_clean = bool((d / "cleaned.txt").exists() or (d / "cleaned.json").exists())
        self.b_clean.setEnabled(can_ai)
        self.b_summarize.setEnabled(can_ai and has_clean)
        self.seg.set_enabled(1, has_clean); self.seg.set_enabled(2, bool((d / "summary.md").exists())); self.seg.set_enabled(3, bool(rows))

    def _on_done(self, d: Path) -> None:
        meta = store.read_meta(d)
        self.ctrl.identify_voices(d)
        rm = self.settings.refine_model
        if (rm and rm != meta.get("model") and rm in available_models(paths.MODELS_DIR)
                and not meta.get("refined") and not meta.get("refine_pending") and d not in self.ctrl.refiners):
            self.ctrl.start_refine(d, rm)
            return
        self._postprocess(d)

    def _prompt_speaker_names(self, d: Path, unnamed: List[str]) -> None:
        """A new, not-yet-recognised voice was heard in this meeting: show what they said (and let
        the user play back a short clip of their voice) before asking what to call it, so it's
        recognised automatically from now on - a bare timestamp isn't enough to tell who it was."""
        rows, speakers = store.read_rows(d), store.read_speakers(d)
        by_speaker: dict = {}
        for r in rows:
            pid = speakers.get(r.get("seq"))
            if pid in unnamed:
                by_speaker.setdefault(pid, []).append(r)
        title = store.read_meta(d).get("title", d.name)
        pcm_path = d / "raw" / "mic.pcm"
        pcm = np.memmap(pcm_path, dtype=np.int16, mode="r") if pcm_path.exists() else None
        for pid in unnamed:
            turns = by_speaker.get(pid, [])
            if not turns:
                continue
            longest = max(turns, key=lambda r: r["t_end"] - r["t_start"])
            text = " ".join(t.get("text", "") for t in turns[:3]).strip()[:220]
            audio = None
            if pcm is not None:
                a, b = int(longest["t_start"] * RATE), min(int(longest["t_end"] * RATE), len(pcm))
                if b > a:
                    audio = pcm[a:b].astype(np.float32) / 32768.0
            dlg = NameSpeakerDialog(title, fmt_dur(longest["t_start"]), text, audio, self)
            if dlg.exec() and dlg.name():
                voiceid.rename_profile(pid, dlg.name())

    def _postprocess(self, d: Path) -> None:
        if self.settings.auto_postprocess and self._llm().available and not (d / "cleaned.json").exists():
            self.run_clean(d)

    # ================================================================== rendering
    def _render(self, view: ReadingView, rows: List[dict], variant: str, cleaned) -> None:
        p = self.tm.p
        groups = export.group_rows(rows, variant, cleaned, roman=self.settings.tamil_display)
        if groups:
            view.setHtml(transcript_html(groups, p))
        elif self.ctrl.active and self.current == self.ctrl.active:
            view.setHtml(placeholder_html("Listening…", "Text appears here a few seconds after people speak.", p))
        else:
            view.setHtml(placeholder_html("Nothing transcribed yet", "If this recording had speech, it will show up here shortly.", p))

    def _cleaned_text(self, d: Path) -> str:
        """The editable cleaned transcript: the user's saved copy if they've edited it, else freshly rendered."""
        tp = d / "cleaned.txt"
        if tp.exists():
            return tp.read_text(encoding="utf-8")
        cl = store.read_json(d / "cleaned.json")
        rows = store.read_rows(d)
        if cl:
            return export.render_txt(rows, "cleaned", {int(k): v for k, v in cl.items()}, roman=self.settings.tamil_display)
        return ""

    def _load_side_files(self) -> None:
        d = self.current
        if not d:
            return
        self._loading_side = True
        try:
            text = self._cleaned_text(d)
            if self.view_clean.toPlainText() != text:
                self.view_clean.setPlainText(text)
            sp = d / "summary.md"
            smd = sp.read_text(encoding="utf-8") if sp.exists() else ""
            if self.view_sum.toPlainText() != smd:
                self.view_sum.setPlainText(smd)
            self.sum_views.setCurrentIndex(0); self.b_sum_edit.setText("Edit")
            self._render_summary_preview()
            hist = store.read_json(d / "chat.json")
            self._chat_history = hist if isinstance(hist, list) else []
            self._render_chat()
        finally:
            self._loading_side = False

    def _render_chat(self) -> None:
        if self._chat_history:
            self.chat_log.setHtml(chat_html(self._chat_history, self.tm.p))
        else:
            self.chat_log.setHtml(placeholder_html("Ask about this meeting",
                                                    "Ask anything about what was discussed - decisions, action items, who said what.", self.tm.p))

    def _render_summary_preview(self) -> None:
        md = self.view_sum.toPlainText().strip()
        if md:
            self.view_sum_preview.setHtml(markdown_html(md, self.tm.p))
        else:
            self.view_sum_preview.setHtml(placeholder_html("No summary yet",
                                                            "Review the Cleaned tab, then click Summarize.", self.tm.p))

    def _toggle_summary_edit(self) -> None:
        if self.sum_views.currentIndex() == 0:                    # previewing -> switch to edit
            self.sum_views.setCurrentIndex(1); self.b_sum_edit.setText("Preview"); self.view_sum.setFocus()
        else:                                                      # editing -> switch to preview
            self._render_summary_preview(); self.sum_views.setCurrentIndex(0); self.b_sum_edit.setText("Edit")

    def _on_tab(self, i: int) -> None:
        self.views.setCurrentIndex(i)

    def _on_clean_edited(self) -> None:
        if self._loading_side or not self.current:
            return
        self._clean_save_t.start()

    def _save_clean(self) -> None:
        if self.current:
            (self.current / "cleaned.txt").write_text(self.view_clean.toPlainText(), encoding="utf-8")

    def _on_summary_edited(self) -> None:
        if self._loading_side or not self.current:
            return
        self._summary_save_t.start()

    def _save_summary(self) -> None:
        if self.current:
            (self.current / "summary.md").write_text(self.view_sum.toPlainText(), encoding="utf-8")

    # ================================================================== actions
    def _llm(self) -> LLMClient:
        s = self.settings
        return LLMClient(s.llm_base_url, s.llm_api_key, s.llm_model, s.llm_verify_tls)

    def _need_llm(self) -> bool:
        if self._llm().available:
            return True
        if QMessageBox.question(self, "LLM not configured", "Enter your company LLM base URL and API key in Settings first.\nOpen Settings now?") == QMessageBox.StandardButton.Yes:
            self.open_settings()
        return self._llm().available

    def _run_job(self, fn, ok, label_text: str, progress_text: Optional[str] = None) -> None:
        j = Job(fn, self); self._jobs.append(j)
        prog = progress_text or label_text
        self._job_text = label_text; self.job_indicator.start(f"{prog}…"); self.refresh()

        def end():
            self._job_text = ""; self.job_indicator.stop(); self.refresh()
        j.progress.connect(lambda a, b: (setattr(self, "_job_text", f"{label_text} {a}/{b}"), self.job_indicator.set_text(f"{prog}… {a}/{b}")))
        j.done.connect(lambda r: (end(), self._toast(f"{label_text} complete"), ok(r)))
        j.failed.connect(lambda e: (end(), QMessageBox.warning(self, label_text, f"Could not complete:\n{e}")))
        j.finished.connect(lambda: self._jobs.remove(j) if j in self._jobs else None)
        j.start()

    def run_clean(self, d: Optional[Path] = None) -> None:
        d = d if isinstance(d, Path) else self.current
        if not d or not self._need_llm():
            return
        rows = store.read_rows(d); client = self._llm(); mode = self.settings.tamil_display

        def work(progress):
            if mode == "english":
                out = client.clean_lines([export._text(r, "raw", None, "english") for r in rows], progress,
                                         alts=[(r.get("text", "") if r.get("lang") == "ta" else "") for r in rows], english=True)
            else:
                out = client.clean_lines([r["text"] for r in rows], progress, alts=[r.get("alt", "") for r in rows], roman=(mode == "roman"))
            cleaned = {str(r["seq"]): t for r, t in zip(rows, out)}
            (d / "cleaned.json").write_text(json.dumps(cleaned, ensure_ascii=False), encoding="utf-8")
            text_path = d / "cleaned.txt"
            if not text_path.exists():            # seed the editable copy once; never clobber a manual edit
                text_path.write_text(export.render_txt(rows, "cleaned", {int(k): v for k, v in cleaned.items()}, roman=mode), encoding="utf-8")
            return True
        self._run_job(work, lambda _: (self._load_side_files(), self.seg.set_current(1)) if d == self.current else None, "Cleaning up")

    def run_summary(self, d: Optional[Path] = None) -> None:
        d = d if isinstance(d, Path) else self.current
        if not d or not self._need_llm():
            return
        client = self._llm(); text = self._cleaned_text(d)
        if not text.strip():
            QMessageBox.information(self, "Nothing to summarize", "There is no transcript text yet."); return

        def work(progress):
            md = client.summarize(text, progress)
            (d / "summary.md").write_text(md, encoding="utf-8"); return md
        self._run_job(work, lambda _: (self._load_side_files(), self.seg.set_current(2)) if d == self.current else None, "Summarising")

    def _ask_context(self, d: Path) -> str:
        text = self._cleaned_text(d) or export.render_txt(store.read_rows(d), roman=self.settings.tamil_display)
        sp = d / "summary.md"
        if sp.exists():
            text += "\n\n## Summary\n" + sp.read_text(encoding="utf-8")
        return text

    def _ask_question(self, q: str) -> None:
        d = self.current
        q = q.strip()
        if not d or not q or not self._need_llm():
            return
        client = self._llm(); context = self._ask_context(d)
        history = list(self._chat_history)                   # local snapshot: isolated from whatever meeting is current later
        convo = history + [{"role": "user", "content": q}]
        self.composer.clear()
        self._chat_history = convo; self._render_chat()

        def work(progress):
            return client.ask(context, history, q)

        def ok(answer):
            full = convo + [{"role": "assistant", "content": answer}]
            (d / "chat.json").write_text(json.dumps(full, ensure_ascii=False), encoding="utf-8")
            if d == self.current:                             # user may have switched meetings while this was in flight
                self._chat_history = full; self._render_chat()
        self._run_job(work, ok, "Asking", progress_text="Thinking")

    def retranscribe(self, model: str) -> None:
        d = self.current
        if not d:
            return
        if QMessageBox.question(self, "Re-transcribe", f"Re-run speech recognition with “{model}”?\nThe previous transcript is kept as a backup file.") == QMessageBox.StandardButton.Yes:
            self._finalized.discard(d); self.ctrl.retranscribe(d, model); self._row_count = -1

    def import_audio(self) -> None:
        f, _ = QFileDialog.getOpenFileName(self, "Import audio or video", "", "Audio/Video (*.m4a *.mp3 *.wav *.mp4 *.mkv *.webm *.ogg *.flac *.aac *.wma *.opus);;All files (*)")
        if not f:
            return
        if not self._models_ok():
            self._first_run_check(); return
        d = self.ctrl.import_file(f, on_error=lambda e: None)
        self.select_meeting(d)

    def _current_text(self, variant: str) -> str:
        d = self.current
        if not d:
            return ""
        if variant == "cleaned":
            text = self._cleaned_text(d)
            if text:
                return text
        return export.render_txt(store.read_rows(d), roman=self.settings.tamil_display)

    def copy_text(self) -> None:
        if not self.current:
            return
        QGuiApplication.clipboard().setText(self._current_text("cleaned" if self.seg.current() == 1 else "raw"))
        self._toast("Transcript copied")

    def save_as(self) -> None:
        d = self.current
        if not d:
            return
        self._flush_edits()                       # make sure cleaned.txt / summary.md are current before export
        meta = store.read_meta(d); rows = store.read_rows(d)
        cl = store.read_json(d / "cleaned.json"); cleaned = {int(k): v for k, v in cl.items()} if cl else None
        variant = "cleaned" if (cleaned and self.seg.current() == 1) else "raw"
        sp = d / "summary.md"; summary = sp.read_text(encoding="utf-8") if sp.exists() else ""
        safe = "".join(ch for ch in meta.get("title", "meeting") if ch not in '\\/:*?"<>|').strip() or "meeting"
        f, flt = QFileDialog.getSaveFileName(self, "Save transcript", str(Path.home() / "Documents" / safe),
                                             "Text (*.txt);;Markdown with summary (*.md);;Subtitles (*.srt)")
        if not f:
            return
        ext = Path(f).suffix.lower() or (".md" if "md" in flt else ".srt" if "srt" in flt else ".txt")
        if not Path(f).suffix:
            f += ext
        rm = self.settings.tamil_display
        # shortcut: .md/.srt export still derives "cleaned" text from the per-line cleaned.json, so manual edits
        # made in the Cleaned box (a single free-text field) aren't reflected there - only in Copy and .txt export.
        body = (export.render_md(rows, meta, variant, cleaned, summary, roman=rm) if ext == ".md" else
                export.render_srt(rows, variant, cleaned, roman=rm) if ext == ".srt" else
                (self._cleaned_text(d) if variant == "cleaned" else export.render_txt(rows, roman=rm)))
        enc = "utf-8-sig" if ext != ".md" else "utf-8"
        Path(f).write_text(body, encoding=enc)
        (d / "exports").mkdir(exist_ok=True)
        (d / "exports" / f"{safe}{ext}").write_text(body, encoding=enc)
        self._toast(f"Saved {Path(f).name}")

    def open_folder(self) -> None:
        if self.current:
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(self.current)))

    def closeEvent(self, e) -> None:
        self._flush_edits()
        if self.ctrl.active:
            r = QMessageBox.question(self, "Recording in progress", "A meeting is still being recorded. Stop recording and exit?")
            if r != QMessageBox.StandardButton.Yes:
                e.ignore(); return
            self.ctrl.stop_live()
        e.accept()
