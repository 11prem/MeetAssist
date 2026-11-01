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


def _voice_env(tmp_path, monkeypatch):
    """Isolated data dirs + a meeting 'Weekly sync' in which voice p1 (unnamed) speaks 2s-5s; returns (app, meeting dir)."""
    import json
    import numpy as np
    from meetassist import paths, store, voiceid
    monkeypatch.setattr(paths, "DATA_DIR", tmp_path); monkeypatch.setattr(paths, "SETTINGS_FILE", tmp_path / "settings.json")
    monkeypatch.setattr(paths, "MEETINGS_DIR", tmp_path / "m"); monkeypatch.setattr(paths, "VOICES_FILE", tmp_path / "voices.json")
    from PySide6.QtWidgets import QApplication
    app = QApplication.instance() or QApplication([])
    theme.init()
    d = store.new_meeting("Weekly sync")
    (d / "journal.jsonl").write_text(encoding="utf-8", data=json.dumps({"seq": 1, "track": "mic", "t_start": 2, "t_end": 5, "text": "hello team", "lang": "en"}) + "\n")
    (d / "raw" / "mic.pcm").write_bytes(np.zeros(16000 * 6, np.int16).tobytes())
    store.write_speakers(d, {1: "p1"})
    voiceid.save_profiles([{"id": "p1", "name": None, "centroid": [1.0, 0.0, 0.0], "count": 1}])
    return app, d


def test_finishing_a_meeting_never_pops_up_the_name_voice_dialog(tmp_path, monkeypatch):
    """Regression: an unnamed voice used to open the naming dialog after every transcription / re-transcribe / refine."""
    app, d = _voice_env(tmp_path, monkeypatch)
    from meetassist.settings import Settings
    from meetassist.ui.main_window import MainWindow
    from meetassist.ui.name_speaker_dialog import NameSpeakerDialog
    opened = []
    monkeypatch.setattr(NameSpeakerDialog, "exec", lambda self: opened.append(1) or 0)
    s = Settings(); s.ack_notice = True; s.voice_enroll_asked = True      # else the one-time "learn your voice" question box blocks the test
    w = MainWindow(s); w.show(); w.select_meeting(d)
    w.ctrl.voiceid_done = [(d, ["p1"])]                        # what a finished voice-id pass used to queue for the GUI
    w.refresh(); app.processEvents()
    assert not opened
    w.close()


def test_voices_page_shows_where_each_voice_was_heard_and_names_it_on_click(tmp_path, monkeypatch):
    app, d = _voice_env(tmp_path, monkeypatch)
    from PySide6.QtCore import QCoreApplication, QEvent
    from PySide6.QtWidgets import QPushButton
    from meetassist import voiceid
    from meetassist.settings import Settings
    from meetassist.ui import settings_dialog
    dlg = settings_dialog.SettingsDialog(Settings())
    rows = lambda: [b for b in dlg.voices_card.findChildren(QPushButton) if b.property("variant") == "side"]
    assert len(rows()) == 1
    assert "Unnamed voice" in rows()[0].text() and "Weekly sync" in rows()[0].text()

    seen = {}
    class FakeDialog:                                          # stands in for the modal naming dialog
        def __init__(self, title, when, text, audio, parent=None, name=""):
            seen.update(title=title, when=when, text=text, has_audio=audio is not None and len(audio) == 3 * 16000, name=name)
        def exec(self): return 1
        def name(self): return "Priya"
    monkeypatch.setattr(settings_dialog, "NameSpeakerDialog", FakeDialog)
    rows()[0].click()
    assert seen == {"title": "Weekly sync", "when": "0:02", "text": "hello team", "has_audio": True, "name": ""}
    assert voiceid.profile_names() == {"p1": "Priya"}
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    assert len(rows()) == 1 and "Priya" in rows()[0].text()


def test_finished_meeting_with_no_speech_does_not_crash(tmp_path, monkeypatch):
    """worker.json has rtf=null when there was 0 s of speech; formatting it used to raise TypeError on launch/refresh."""
    import json
    app, d = _voice_env(tmp_path, monkeypatch)
    (d / "journal.jsonl").write_text("", encoding="utf-8")
    (d / "worker.json").write_text(json.dumps({"state": "done", "rtf": None, "audio_s": 31}), encoding="utf-8")
    from meetassist.settings import Settings
    from meetassist.ui.main_window import MainWindow
    s = Settings(); s.ack_notice = True
    w = MainWindow(s); w.show(); w.select_meeting(d); w.refresh(); app.processEvents()
    assert "0.00× real-time" in w.detail.text()
    w.close()


def test_play_sample_uses_winsound_not_the_default_player(monkeypatch):
    import winsound
    import numpy as np
    from PySide6.QtWidgets import QApplication
    from meetassist.ui.name_speaker_dialog import NameSpeakerDialog
    app = QApplication.instance() or QApplication([])
    calls = []
    monkeypatch.setattr(winsound, "PlaySound", lambda *a: calls.append(a))
    dlg = NameSpeakerDialog("T", "0:01", "hi", np.zeros(1600, np.float32))
    dlg._play()
    assert calls == [(str(dlg._clip_path), winsound.SND_FILENAME | winsound.SND_ASYNC)]
    dlg._clip_path.unlink()


def test_play_sample_failure_is_shown_to_the_user(monkeypatch):
    import winsound
    import numpy as np
    from PySide6.QtWidgets import QApplication, QMessageBox
    from meetassist.ui.name_speaker_dialog import NameSpeakerDialog
    app = QApplication.instance() or QApplication([])
    def no_device(*a):
        raise RuntimeError("Failed to play sound")
    shown = []
    monkeypatch.setattr(winsound, "PlaySound", no_device)
    monkeypatch.setattr(QMessageBox, "warning", lambda parent, title, text: shown.append(text))
    dlg = NameSpeakerDialog("T", "0:01", "hi", np.zeros(1600, np.float32))
    dlg._play()
    assert len(shown) == 1 and "Failed to play sound" in shown[0]
    dlg._clip_path.unlink()


def test_refine_done_re_identifies_voices_and_merged_names_show_one_entry(tmp_path, monkeypatch):
    app, d = _voice_env(tmp_path, monkeypatch)
    from PySide6.QtCore import QCoreApplication, QEvent
    from PySide6.QtWidgets import QPushButton
    from meetassist import voiceid
    from meetassist.settings import Settings
    from meetassist.ui import settings_dialog
    from meetassist.ui.main_window import MainWindow
    s = Settings(); s.ack_notice = True; s.voice_enroll_asked = True
    w = MainWindow(s); w.show(); w.select_meeting(d)
    seen = []
    monkeypatch.setattr(w.ctrl, "identify_voices", lambda r: seen.append(r))
    w.ctrl.refine_done.append(d)                       # a refinement just swapped in a new journal (speakers.json was dropped)
    w.refresh(); app.processEvents()
    assert seen == [d]
    w.close()

    voiceid.save_profiles([{"id": "p1", "name": "Priya", "centroid": [1.0, 0.0, 0.0], "count": 1},
                           {"id": "p2", "name": None, "centroid": [1.0, 0.0, 0.0], "count": 1}])
    dlg = settings_dialog.SettingsDialog(Settings())
    rows = lambda: [b for b in dlg.voices_card.findChildren(QPushButton) if b.property("variant") == "side"]
    assert len(rows()) == 2
    class FakeDialog:
        def __init__(self, *a, **k): pass
        def exec(self): return 1
        def name(self): return "priya"
    monkeypatch.setattr(settings_dialog, "NameSpeakerDialog", FakeDialog)
    rows()[1].click()
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    assert len(rows()) == 1 and "Priya" in rows()[0].text()


def test_record_voice_thread_reports_why_it_failed(tmp_path, monkeypatch):
    """#2: a capture failure used to kill the thread silently; the receivers now get (False, reason)."""
    import numpy as np
    from meetassist import voiceid
    from meetassist.ui.settings_dialog import _RecordVoice

    class Ctrl:
        def record_voice_sample(self, seconds): raise RuntimeError("microphone unavailable (denied)")
    got = []
    t = _RecordVoice(Ctrl(), 0.1); t.done.connect(lambda ok, msg: got.append((ok, msg))); t.run()
    assert got == [(False, "microphone unavailable (denied)")]

    class Quiet:
        def record_voice_sample(self, seconds): return np.zeros(10, np.float32)
    monkeypatch.setattr(voiceid, "enroll_me", lambda a: False)
    got.clear()
    t = _RecordVoice(Quiet(), 0.1); t.done.connect(lambda ok, msg: got.append((ok, msg))); t.run()
    assert got[0][0] is False and got[0][1]
    monkeypatch.setattr(voiceid, "enroll_me", lambda a: True)
    got.clear()
    t = _RecordVoice(Quiet(), 0.1); t.done.connect(lambda ok, msg: got.append((ok, msg))); t.run()
    assert got == [(True, "")]
