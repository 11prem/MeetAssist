"""Dialog shown when a new, not-yet-recognised voice is heard. Naming a bare timestamp ("a voice
was heard around 00:03:12") isn't enough to tell who it was in a multi-person meeting, so this shows
what that person actually said and lets the user play back a short clip of their voice before naming them."""
from __future__ import annotations

import os
import tempfile
import wave
from pathlib import Path
from typing import Optional

import numpy as np
from PySide6.QtWidgets import QDialog, QDialogButtonBox, QLineEdit, QPushButton, QVBoxLayout

from .theme import SP
from .widgets import label


class NameSpeakerDialog(QDialog):
    def __init__(self, title: str, when: str, text: str, audio: Optional[np.ndarray], parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Name this voice")
        self.setMinimumWidth(420)
        self._clip_path: Optional[Path] = None
        v = QVBoxLayout(self)
        v.setContentsMargins(SP["xl"], SP["xl"], SP["xl"], SP["xl"]); v.setSpacing(SP["md"])
        v.addWidget(label(f"A new voice was heard in “{title}” around {when}.", "callout", wrap=True))
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
        self.name_edit = QLineEdit()
        self.name_edit.setPlaceholderText("Name")
        v.addWidget(self.name_edit)
        box = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        box.accepted.connect(self.accept)
        box.rejected.connect(self.reject)
        v.addWidget(box)
        self.name_edit.setFocus()

    def _play(self) -> None:
        if self._clip_path:
            os.startfile(str(self._clip_path))          # Windows-only app; opens with the default audio player

    @staticmethod
    def _write_clip(audio: np.ndarray) -> Path:
        # shortcut: the temp clip is never deleted (the external player may still be reading it
        # after this dialog closes); if these accumulate, clean up meetassist_voice_*.wav in %TEMP%.
        fd, path = tempfile.mkstemp(suffix=".wav", prefix="meetassist_voice_")
        os.close(fd)
        with wave.open(path, "wb") as w:
            w.setnchannels(1); w.setsampwidth(2); w.setframerate(16000)
            w.writeframes((np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16).tobytes())
        return Path(path)

    def name(self) -> str:
        return self.name_edit.text().strip()
