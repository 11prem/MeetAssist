import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
import pytest
from meetassist.ui import theme


@pytest.fixture(autouse=True)
def _no_first_run_popup(monkeypatch):
    """A MainWindow from an earlier test fires its 300 ms first-run timer during a later one; the modal question then blocks forever."""
    from meetassist.ui.main_window import MainWindow
    monkeypatch.setattr(MainWindow, "_first_run_check", lambda self: None)


PAIRS = [  # (foreground, background, minimum WCAG ratio) - checked in BOTH themes
    ("text", "bg", 7.0), ("text2", "bg", 7.0), ("text3", "bg", 4.5), ("text", "sidebar", 7.0), ("text3", "sidebar", 4.5),
    ("text", "surface", 7.0), ("accent", "bg", 4.5), ("on_accent", "accent_fill", 4.5), ("accent", "bg", 4.5), ("success", "bg", 4.5), ("text", "fill", 7.0), ("text3", "fill", 4.5), ("you", "bg", 4.5), ("remote", "bg", 4.5),
    ("text", "accent_tint", 7.0), ("text", "warning_tint", 7.0), ("warning", "warning_tint", 4.5), ("text", "record_tint", 7.0),
]

@pytest.mark.parametrize("pal", [theme.LIGHT, theme.DARK], ids=["light", "dark"])
def test_palette_contrast_wcag(pal):
    for fg, bg, need in PAIRS:
        got = theme.contrast(getattr(pal, fg), getattr(pal, bg))
        assert got >= need, f"{pal.name}: {fg} on {bg} = {got:.2f} < {need}"

def test_window_builds_and_refreshes_in_both_themes(tmp_path, monkeypatch):
    from meetassist import paths
    monkeypatch.setattr(paths, "DATA_DIR", tmp_path); monkeypatch.setattr(paths, "SETTINGS_FILE", tmp_path / "settings.json"); monkeypatch.setattr(paths, "MEETINGS_DIR", tmp_path / "m")
    from PySide6.QtWidgets import QApplication
    app = QApplication.instance() or QApplication([])
    theme.init()
    from meetassist import store
    from meetassist.settings import Settings
    from meetassist.ui.main_window import MainWindow
    import json
    d = store.new_meeting("Smoke test"); (d / "journal.jsonl").write_text(encoding="utf-8", data=json.dumps(
        {"seq": 1, "track": "mic", "t_start": 1, "t_end": 3, "text": "இது ஒரு டிரிப்", "lang": "ta", "en": "This is for a trip"}, ensure_ascii=False) + "\n")
    s = Settings(); s.ack_notice = True
    w = MainWindow(s); w.show()
    for mode in ("light", "dark", "system"):
        w.tm.set_mode(mode); w.select_meeting(d); w.refresh(); app.processEvents()
        assert "This is for a trip" in w.view_raw.toPlainText()
    w.seg.set_current(2); w.seg.set_current(0)
    w._set_display("roman"); assert "idhu" in w.view_raw.toPlainText().lower() or "oru" in w.view_raw.toPlainText().lower()
    w.close()

def test_clean_up_progress_belongs_to_its_own_meeting(tmp_path, monkeypatch):
    """The "Cleaning up…" indicator must show only on the meeting being cleaned, not follow the user to others."""
    import json, threading, time
    from meetassist import paths
    monkeypatch.setattr(paths, "DATA_DIR", tmp_path); monkeypatch.setattr(paths, "SETTINGS_FILE", tmp_path / "settings.json"); monkeypatch.setattr(paths, "MEETINGS_DIR", tmp_path / "m")
    from PySide6.QtWidgets import QApplication
    app = QApplication.instance() or QApplication([])
    theme.init()
    from meetassist import store
    from meetassist.settings import Settings
    from meetassist.ui.main_window import MainWindow

    def meeting(title, text):
        d = store.new_meeting(title)
        (d / "journal.jsonl").write_text(encoding="utf-8", data=json.dumps({"seq": 1, "track": "mic", "t_start": 1, "t_end": 3, "text": text, "lang": "en"}) + "\n")
        return d
    a, b = meeting("A", "first meeting"), meeting("B", "second meeting")
    release = {"first meeting": threading.Event(), "second meeting": threading.Event()}

    class SlowLLM:                                              # holds each clean-up in flight until the test lets it finish
        available = True
        def clean_lines(self, lines, progress, **_):
            release[lines[0]].wait(5)
            return lines
    monkeypatch.setattr(MainWindow, "_llm", lambda self: SlowLLM())

    s = Settings(); s.ack_notice = True; s.tamil_display = "tamil"
    w = MainWindow(s); w.show()

    def pump(until=lambda: False, timeout=5.0):
        end = time.time() + timeout
        while not until() and time.time() < end:
            app.processEvents(); time.sleep(0.01)
        app.processEvents()
    def shown(): app.processEvents(); return not w.job_indicator.isHidden()

    w.select_meeting(a); w.run_clean()
    assert shown(), "indicator should show on the meeting being cleaned"
    w.select_meeting(b)
    assert not shown(), "indicator must not appear on a different meeting"
    w.run_clean()
    assert shown()
    release["first meeting"].set(); pump(lambda: len(w._jobs) == 1)
    assert shown(), "finishing A's job must not hide B's still-running indicator"
    w.select_meeting(a)
    assert not shown(), "A is done"
    w.select_meeting(b)
    assert shown()
    release["second meeting"].set(); pump(lambda: not w._jobs)
    assert not shown()
    w.close()


def _meeting_env(tmp_path, monkeypatch):
    """Isolated data dirs + a meeting 'Weekly sync' with one line of speech at 2s-5s; returns (app, meeting dir)."""
    import json
    from meetassist import paths, store
    monkeypatch.setattr(paths, "DATA_DIR", tmp_path); monkeypatch.setattr(paths, "SETTINGS_FILE", tmp_path / "settings.json")
    monkeypatch.setattr(paths, "MEETINGS_DIR", tmp_path / "m")
    from PySide6.QtWidgets import QApplication
    app = QApplication.instance() or QApplication([])
    theme.init()
    d = store.new_meeting("Weekly sync")
    (d / "journal.jsonl").write_text(encoding="utf-8", data=json.dumps({"seq": 1, "track": "mic", "t_start": 2, "t_end": 5, "text": "hello team", "lang": "en"}) + "\n")
    return app, d


def test_finished_meeting_with_no_speech_does_not_crash(tmp_path, monkeypatch):
    """worker.json has rtf=null when there was 0 s of speech; formatting it used to raise TypeError on launch/refresh."""
    import json
    app, d = _meeting_env(tmp_path, monkeypatch)
    (d / "journal.jsonl").write_text("", encoding="utf-8")
    (d / "worker.json").write_text(json.dumps({"state": "done", "rtf": None, "audio_s": 31}), encoding="utf-8")
    from meetassist.settings import Settings
    from meetassist.ui.main_window import MainWindow
    s = Settings(); s.ack_notice = True
    w = MainWindow(s); w.show(); w.select_meeting(d); w.refresh(); app.processEvents()
    assert "0.00× real-time" in w.detail.text()
    w.close()
