"""Dialog opened from Settings > Voices to name a voice (never shown on its own). Naming a bare timestamp
("a voice was heard around 00:03:12") isn't enough to tell who it was in a multi-person meeting, so this
shows what that person actually said and lets the user play back a short clip of their voice before naming them."""
from __future__ import annotations

import os
import tempfile
import wave
import winsound
from pathlib import Path
from typing import Optional

import numpy as np
from PySide6.QtWidgets import QDialog, QDialogButtonBox, QLineEdit, QMessageBox, QPushButton, QVBoxLayout

from .theme import SP
from .widgets import label


class NameSpeakerDialog(QDialog):
    def __init__(self, title: Optional[str], when: str, text: str, audio: Optional[np.ndarray], parent=None, name: str = "") -> None:
        super().__init__(parent)
        self.setWindowTitle("Name this voice")
        self.setMinimumWidth(420)
        self._clip_path: Optional[Path] = None
        v = QVBoxLayout(self)
        v.setContentsMargins(SP["xl"], SP["xl"], SP["xl"], SP["xl"]); v.setSpacing(SP["md"])
        v.addWidget(label(f"This voice was heard in “{title}” around {when}." if title
                          else "This voice isn't in any saved meeting any more.", "callout", wrap=True))
        if text:
            quote = label(f"“{text}”", "body2", wrap=True)
            quote.setStyleSheet(quote.styleSheet() + "font-style: italic;")
            v.addWidget(quote)
        if audio is not None and len(audio) > 0:
            self._clip_path = self._write_clip(audio)
            play = QPushButton("▶  Play sample")
            play.clicked.connect(self._play)
            v.addWidget(play)
        v.addWidget(label("Who is this? (leave blank to decide later)", "footnote", wrap=True))
        self.name_edit = QLineEdit(name)
        self.name_edit.setPlaceholderText("Name")
        v.addWidget(self.name_edit)
        box = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        box.accepted.connect(self.accept)
        box.rejected.connect(self.reject)
        v.addWidget(box)
        self.name_edit.setFocus()

    def _play(self) -> None:
        if self._clip_path:
            # in-app playback: corporate laptops often have no default WAV player for os.startfile
            try:
                winsound.PlaySound(str(self._clip_path), winsound.SND_FILENAME | winsound.SND_ASYNC)
            except (RuntimeError, OSError) as e:        # e.g. no audio device: say so rather than failing silently
                QMessageBox.warning(self, "Can't play sample", str(e))

    @staticmethod
    def _write_clip(audio: np.ndarray) -> Path:
        # shortcut: the temp clip is never deleted (async playback may still be reading it
        # after this dialog closes); if these accumulate, clean up meetassist_voice_*.wav in %TEMP%.
        fd, path = tempfile.mkstemp(suffix=".wav", prefix="meetassist_voice_")
        os.close(fd)
        with wave.open(path, "wb") as w:
            w.setnchannels(1); w.setsampwidth(2); w.setframerate(16000)
            w.writeframes((np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16).tobytes())
        return Path(path)

    def name(self) -> str:
        return self.name_edit.text().strip()
