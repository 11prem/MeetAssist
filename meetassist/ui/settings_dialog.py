"""Settings - Apple-style: navigation list on the left, grouped cards with switches on the right."""
from __future__ import annotations

import json
import os
import subprocess
import sys

from PySide6.QtCore import QThread, QUrl, Qt, Signal
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (QComboBox, QDialog, QFrame, QHBoxLayout, QLineEdit, QListWidget, QMessageBox,
                               QPlainTextEdit, QPushButton, QScrollArea, QStackedWidget, QVBoxLayout, QWidget)

from .. import __version__, paths, voiceid
from ..llm import LLMClient
from ..settings import Settings
from ..transcriber import MODEL_DIRS, available_models
from . import theme
from .theme import SP
from .widgets import Segmented, Switch, label

MODEL_HELP = {"large-v3-turbo": "most accurate, slowest", "medium": "balanced", "small": "fastest, great for live meetings"}
LANGS = [("hybrid", "Tamil + English + Tanglish (recommended)"),
         ("en", "English only"),
         ("ta", "Tamil only (fastest)"),
         ("whisper", "Whisper only (Tamil in English letters)")]


class _Ping(QThread):
    ok = Signal(str)
    bad = Signal(str)

    def __init__(self, client: LLMClient) -> None:
        super().__init__()
        self.client = client

    def run(self) -> None:
        try:
            self.ok.emit(self.client.ping())
        except Exception as e:
            self.bad.emit(str(e))


class _RecordVoice(QThread):
    done = Signal(bool)

    def __init__(self, ctrl, seconds: float = 10.0) -> None:
        super().__init__()
        self.ctrl = ctrl
        self.seconds = seconds

    def run(self) -> None:
        audio = self.ctrl.record_voice_sample(self.seconds)
        self.done.emit(voiceid.enroll_me(audio))


def _row(title: str, desc: str, control: QWidget, stacked: bool = False) -> QWidget:
    w = QWidget()
    lay = QVBoxLayout(w) if stacked else QHBoxLayout(w)
    lay.setContentsMargins(SP["lg"], SP["md"], SP["lg"], SP["md"])
    lay.setSpacing(SP["sm"] if stacked else SP["lg"])
    txt = QVBoxLayout(); txt.setSpacing(2)
    txt.addWidget(label(title, "callout"))
    if desc:
        txt.addWidget(label(desc, "footnote", wrap=True))
    if stacked:
        lay.addLayout(txt); lay.addWidget(control)
    else:
        lay.addLayout(txt, 1); lay.addWidget(control, 0, Qt.AlignmentFlag.AlignVCenter)
    return w


class _Card(QFrame):
    def __init__(self) -> None:
        super().__init__()
        self.setObjectName("card")
        self._lay = QVBoxLayout(self); self._lay.setContentsMargins(0, 0, 0, 0); self._lay.setSpacing(0)
        self._n = 0

    def add(self, w: QWidget) -> QWidget:
        if self._n:
            sep = QFrame(); sep.setObjectName("hsep"); self._lay.addWidget(sep)
        self._lay.addWidget(w)
        self._n += 1
        return w


def _page(title: str, subtitle: str = "") -> tuple:
    w = QWidget(); w.setObjectName("page")
    v = QVBoxLayout(w); v.setContentsMargins(SP["xl"], SP["xl"], SP["xl"], SP["xl"]); v.setSpacing(SP["lg"])
    v.addWidget(label(title, "title"))
    if subtitle:
        v.addWidget(label(subtitle, "body2", wrap=True))
    return w, v


class SettingsDialog(QDialog):
    def __init__(self, s: Settings, ctrl=None, parent=None) -> None:
        super().__init__(parent)
        self.s = s
        self.ctrl = ctrl
        self.setObjectName("settings")
        self.setWindowTitle("Settings")
        self.resize(840, 660)
        self.setMinimumSize(740, 540)
        root = QVBoxLayout(self); root.setContentsMargins(0, 0, 0, 0); root.setSpacing(0)
        body = QHBoxLayout(); body.setSpacing(0); root.addLayout(body, 1)
        self.nav = QListWidget(); self.nav.setObjectName("nav"); self.nav.setFixedWidth(210); self.nav.setAccessibleName("Settings sections")
        for n in ("Speech", "AI assistant", "Audio", "Voices", "Appearance", "About"):
            self.nav.addItem(n)
        self.stack = QStackedWidget()
        body.addWidget(self.nav); body.addWidget(self.stack, 1)
        self.nav.currentRowChanged.connect(self.stack.setCurrentIndex)
        for builder in (self._speech, self._ai, self._audio, self._voices, self._appearance, self._about):
            page = builder()
            sc = QScrollArea(); sc.setWidgetResizable(True); sc.setWidget(page); sc.setFrameShape(QFrame.Shape.NoFrame)
            self.stack.addWidget(sc)
        self.nav.setCurrentRow(0)

        foot = QFrame(); foot.setObjectName("page")
        fl = QHBoxLayout(foot); fl.setContentsMargins(SP["xl"], SP["md"], SP["xl"], SP["md"]); fl.addStretch()
        cancel = QPushButton("Cancel"); cancel.clicked.connect(self.reject)
        save = QPushButton("Save"); save.setProperty("variant", "primary"); save.setDefault(True); save.clicked.connect(self._save)
        fl.addWidget(cancel); fl.addWidget(save)
        sep = QFrame(); sep.setObjectName("hsep"); root.addWidget(sep); root.addWidget(foot)
        theme.init().changed.connect(lambda: self.setStyleSheet(theme.build_qss(theme.P())))
        self.setStyleSheet(theme.build_qss(theme.P()))

    # ------------------------------------------------------------------ pages
    def _speech(self) -> QWidget:
        s = self.s
        page, v = _page("Speech", "Speech recognition runs entirely on this computer.")
        card = _Card()
        self.lang = QComboBox()
        for k, t in LANGS:
            self.lang.addItem(t, k)
        self.lang.setCurrentIndex(max(self.lang.findData(s.language), 0)); self.lang.setMinimumWidth(280)
        card.add(_row("Language", "Tamil and Tanglish can be shown as English, Tanglish or Tamil script from the transcript toolbar.", self.lang))
        self.mdl = QComboBox(); self.mdl.setMinimumWidth(280)
        avail = available_models(paths.MODELS_DIR)
        for k in MODEL_DIRS:
            if k in avail or k == s.model_name:
                self.mdl.addItem(f"{k}  -  {MODEL_HELP[k]}" + ("" if k in avail else "  (not installed)"), k)
        self.mdl.setCurrentIndex(max(self.mdl.findData(s.model_name), 0))
        card.add(_row("English model", "Whisper model used for English speech.", self.mdl))
        self.refine = QComboBox(); self.refine.addItem("Off (recommended)", ""); self.refine.setMinimumWidth(280)
        extra = [k for k in MODEL_DIRS if k != "small" and k in avail]
        for k in extra:
            self.refine.addItem(f"Re-run English with {k} after the meeting", k)
        self.refine.setCurrentIndex(max(self.refine.findData(s.refine_model), 0))
        if extra:
            card.add(_row("Extra accuracy pass", "A second, slower pass that replaces the draft when finished.", self.refine))
        v.addWidget(card)
        self.voc = QPlainTextEdit(s.vocabulary); self.voc.setFixedHeight(72)
        self.voc.setPlaceholderText("Python, LLM, Kubernetes, API, Docker"); self.voc.setAccessibleName("Vocabulary hints")
        card2 = _Card(); card2.add(_row("Vocabulary hints", "Names and terms to spell correctly, separated by commas.", self.voc, stacked=True))
        v.addWidget(card2); v.addStretch()
        return page

    def _ai(self) -> QWidget:
        s = self.s
        page, v = _page("AI assistant", "Optional. Used only when you ask for a clean-up or summary. This is the only connection MeetAssist ever makes.")
        card = _Card()
        self.url = QLineEdit(s.llm_base_url); self.url.setPlaceholderText("https://llm.company.internal/v1"); self.url.setAccessibleName("LLM base URL")
        self.key = QLineEdit(s.llm_api_key); self.key.setEchoMode(QLineEdit.EchoMode.Password); self.key.setPlaceholderText("API key"); self.key.setAccessibleName("API key")
        self.model = QLineEdit(s.llm_model); self.model.setAccessibleName("Model name")
        card.add(_row("Base URL", "", self.url, stacked=True))
        card.add(_row("API key", "Stored encrypted for your Windows account.", self.key, stacked=True))
        card.add(_row("Model name", "", self.model, stacked=True))
        v.addWidget(card)
        card2 = _Card()
        self.tls = Switch(s.llm_verify_tls); self.tls.setAccessibleName("Verify TLS certificate")
        card2.add(_row("Verify TLS certificate", "Turn off only for an internal self-signed certificate.", self.tls))
        self.auto = Switch(s.auto_postprocess); self.auto.setAccessibleName("Clean up automatically")
        card2.add(_row("Clean up automatically", "After every meeting, once the transcript is ready. Review it on the Cleaned "
                                                 "tab, then click Summarize there when you're ready.", self.auto))
        v.addWidget(card2)
        row = QHBoxLayout(); row.setSpacing(SP["md"])
        self.test = QPushButton("Test connection"); self.test.clicked.connect(self._test)
        self.test_lbl = label("", "footnote", wrap=True)
        row.addWidget(self.test); row.addWidget(self.test_lbl, 1); v.addLayout(row)
        v.addStretch()
        return page

    def _audio(self) -> QWidget:
        s = self.s
        page, v = _page("Audio", "What to record during a meeting.")
        card = _Card()
        self.use_mic = Switch(s.record_mic); self.use_mic.setAccessibleName("Record microphone")
        card.add(_row("Microphone", "Your voice, labelled “You”.", self.use_mic))
        self.use_sys = Switch(s.record_system_audio); self.use_sys.setAccessibleName("Record system audio")
        card.add(_row("System audio", "Everyone on Teams, Webex or Zoom, labelled “Remote”.", self.use_sys))
        self.mic = QComboBox(); self.mic.addItem("Windows default", ""); self.mic.setMinimumWidth(280)
        for name in self._mics():
            self.mic.addItem(name, name)
        self.mic.setCurrentIndex(max(self.mic.findData(s.mic_name), 0))
        card.add(_row("Microphone device", "", self.mic))
        v.addWidget(card)
        v.addWidget(label("Tip: a headset gives the cleanest transcript. With laptop speakers the microphone also hears the other side; MeetAssist hides most of that echo.", "footnote", wrap=True))
        v.addStretch()
        return page

    def _voices(self) -> QWidget:
        page, v = _page("Voices", "In-person meetings hear everyone through one microphone, so MeetAssist "
                                   "tells speakers apart by voice instead of assuming the mic is always you.")
        card = _Card()
        self.rec_btn = QPushButton("Record my voice (10s)")
        self.rec_btn.clicked.connect(self._record_my_voice)
        self.rec_lbl = label("Speak normally after clicking; this is a one-time setup.", "footnote", wrap=True)
        card.add(_row("Me", "", self.rec_btn, stacked=False))
        v.addWidget(card)
        v.addWidget(self.rec_lbl)
        self.voices_card = _Card()
        v.addWidget(label("Other recognised voices", "callout"))
        v.addWidget(self.voices_card)
        v.addStretch()
        self._refresh_voices_list()
        return page

    def _refresh_voices_list(self) -> None:
        card = self.voices_card
        while card._lay.count():
            w = card._lay.takeAt(0).widget()
            if w:
                w.deleteLater()
        card._n = 0
        profiles = [p for p in voiceid.load_profiles() if p["id"] != voiceid.ME_ID]
        if not profiles:
            card.add(_row("None yet", "They appear here once MeetAssist hears more than one voice in a meeting.", QWidget()))
            return
        for p in profiles:
            pid = p["id"]
            edit = QLineEdit(p.get("name") or "")
            edit.setPlaceholderText("Not named yet")
            edit.editingFinished.connect(lambda e=edit, pid=pid: e.text().strip() and voiceid.rename_profile(pid, e.text().strip()))
            rm = QPushButton("Remove")
            rm.clicked.connect(lambda _=False, pid=pid: (voiceid.delete_profile(pid), self._refresh_voices_list()))
            row = QWidget(); rl = QHBoxLayout(row)
            rl.setContentsMargins(SP["lg"], SP["sm"], SP["lg"], SP["sm"]); rl.setSpacing(SP["md"])
            rl.addWidget(edit, 1); rl.addWidget(rm)
            card.add(row)

    def _record_my_voice(self) -> None:
        if not self.ctrl:
            return
        self.rec_btn.setEnabled(False)
        self.rec_lbl.setText("Recording - speak normally for 10 seconds...")
        self._rec = _RecordVoice(self.ctrl)
        self._rec.done.connect(self._on_voice_recorded)
        self._rec.start()

    def _on_voice_recorded(self, ok: bool) -> None:
        self.rec_btn.setEnabled(True)
        self.rec_lbl.setText("Saved." if ok else "Could not save - check the microphone in Audio settings, "
                                                  "or that the speaker-id model is installed.")

    def _appearance(self) -> QWidget:
        page, v = _page("Appearance", "MeetAssist follows Windows light and dark mode by default.")
        card = _Card()
        self.theme_seg = Segmented(["System", "Light", "Dark"])
        self.theme_seg.set_current({"system": 0, "light": 1, "dark": 2}.get(self.s.appearance, 0), animate=False)
        self.theme_seg.currentChanged.connect(lambda i: theme.init().set_mode(("system", "light", "dark")[i]))
        card.add(_row("Theme", "", self.theme_seg))
        v.addWidget(card); v.addStretch()
        return page

    def _about(self) -> QWidget:
        page, v = _page("About", f"MeetAssist {__version__}")
        card = _Card()
        card.add(_row("Privacy", "Audio and transcripts never leave this computer. The only network connection is to the company LLM address you enter, and only when you use the AI features.", QWidget(), stacked=True))
        for title, path in (("Models folder", paths.MODELS_DIR), ("Meetings and settings", paths.DATA_DIR), ("Log file", paths.LOG_FILE)):
            b = QPushButton("Open")
            b.clicked.connect(lambda _=False, p=path: QDesktopServices.openUrl(QUrl.fromLocalFile(str(p if p.is_dir() else p.parent))))
            card.add(_row(title, str(path), b))
        v.addWidget(card); v.addStretch()
        return page

    @staticmethod
    def _mics():
        try:
            py = sys.executable
            if py.lower().endswith("pythonw.exe"):
                py = py[:-11] + "python.exe"
            out = subprocess.run([py, "-m", "meetassist.capture_proc", "--list"], capture_output=True, text=True,
                                 timeout=20, cwd=str(paths.APP_ROOT), creationflags=0x08000000,
                                 env={**os.environ, "PYTHONPATH": str(paths.APP_ROOT)}).stdout
            return json.loads(out.strip().splitlines()[-1])
        except Exception:
            return []

    # ------------------------------------------------------------------ actions
    def _test(self) -> None:
        self.test_lbl.setText("Testing…")
        self.test.setEnabled(False)
        c = LLMClient(self.url.text(), self.key.text(), self.model.text(), self.tls.isChecked(), timeout=30, retries=1)
        self._p = _Ping(c)
        p = theme.P()
        self._p.ok.connect(lambda r: (self.test_lbl.setText("Connected"), self.test_lbl.setStyleSheet(f"color: {p.success};"), self.test.setEnabled(True)))
        self._p.bad.connect(lambda e: (self.test_lbl.setText("Could not connect: " + e[:200]), self.test_lbl.setStyleSheet(f"color: {p.record};"), self.test.setEnabled(True)))
        self._p.start()

    def reject(self) -> None:
        theme.init().set_mode(self.s.appearance)          # undo the live theme preview
        super().reject()

    def _save(self) -> None:
        s = self.s
        s.llm_base_url, s.llm_api_key = self.url.text().strip(), self.key.text().strip()
        s.llm_model = self.model.text().strip() or "deepseek-chat"
        s.llm_verify_tls, s.auto_postprocess = self.tls.isChecked(), self.auto.isChecked()
        s.model_name, s.language = self.mdl.currentData(), self.lang.currentData()
        s.vocabulary = self.voc.toPlainText().strip()
        s.refine_model = self.refine.currentData() or ""
        s.record_mic, s.record_system_audio = self.use_mic.isChecked(), self.use_sys.isChecked()
        s.mic_name = self.mic.currentData() or ""
        s.appearance = ("system", "light", "dark")[self.theme_seg.current()]
        if s.language in ("hybrid", "ta") and not (paths.MODELS_DIR / "indic-asr" / "assets" / "encoder.onnx").exists():
            QMessageBox.warning(self, "Tamil model not installed", f"The Tamil speech model is missing from {paths.MODELS_DIR / 'indic-asr'}.")
            return
        if s.model_name not in available_models(paths.MODELS_DIR):
            QMessageBox.warning(self, "Model not installed", f"The '{s.model_name}' model files are not in {paths.MODELS_DIR}.")
            return
        self.accept()
