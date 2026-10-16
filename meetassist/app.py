"""Application entry point:  python -m meetassist.app   (MeetAssist.exe launches exactly this)."""
from __future__ import annotations

import faulthandler
import os
import sys
import traceback

from . import paths


def main() -> int:
    paths.ensure_dirs()
    if sys.executable.lower().endswith("pythonw.exe") or sys.stderr is None:
        log = open(paths.LOG_FILE, "a", encoding="utf-8", buffering=1)      # windowed: no console, log to file
        sys.stdout = sys.stderr = log
        faulthandler.enable(log)
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    from PySide6.QtCore import QLockFile
    from PySide6.QtWidgets import QApplication, QMessageBox
    from .settings import Settings
    from .ui.main_window import MainWindow

    app = QApplication(sys.argv)
    app.setApplicationName("MeetAssist")
    app.setStyle("Fusion")
    from .ui import theme
    theme.init()
    app.setFont(theme.app_font())
    from .ui.icon import make_icon
    app.setWindowIcon(make_icon())
    lock = QLockFile(str(paths.DATA_DIR / "app.lock"))
    lock.setStaleLockTime(5000)
    if not lock.tryLock(100):
        QMessageBox.information(None, "MeetAssist", "MeetAssist is already running.")
        return 0
    sys.excepthook = lambda t, v, tb: traceback.print_exception(t, v, tb)

    w = MainWindow(Settings.load())
    w.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
