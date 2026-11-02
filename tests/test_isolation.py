import os
from pathlib import Path

from meetassist import paths


def test_tests_never_point_at_the_real_user_data():
    """Leaked timers/threads from UI tests once wrote test data into the real data folder - the session must be sandboxed."""
    real = Path(os.environ.get("LOCALAPPDATA", Path.home())) / "MeetAssist"
    for p in (paths.DATA_DIR, paths.MEETINGS_DIR, paths.SETTINGS_FILE, paths.LOG_FILE):
        assert real.resolve() not in [p.resolve(), *p.resolve().parents], p
