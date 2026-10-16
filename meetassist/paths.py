"""Where things live. App folder is read-only-safe; user data goes to %LOCALAPPDATA%."""
from __future__ import annotations
import os, sys
from pathlib import Path

# <app root> = folder that contains `meetassist/`, `models/`, `runtime/` (and MeetAssist.exe when frozen)
if getattr(sys, "frozen", False):
    APP_ROOT = Path(sys.executable).resolve().parent
else:
    APP_ROOT = Path(__file__).resolve().parent.parent

MODELS_DIR = Path(os.environ.get("MEETASSIST_MODELS_DIR", APP_ROOT / "models"))
DATA_DIR = Path(os.environ.get("MEETASSIST_DATA_DIR", Path(os.environ.get("LOCALAPPDATA", Path.home())) / "MeetAssist"))
MEETINGS_DIR = DATA_DIR / "meetings"
SETTINGS_FILE = DATA_DIR / "settings.json"
LOG_FILE = DATA_DIR / "app.log"

def ensure_dirs() -> None:
    MEETINGS_DIR.mkdir(parents=True, exist_ok=True)
