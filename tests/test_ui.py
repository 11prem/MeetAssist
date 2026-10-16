import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
import pytest
from meetassist.ui import theme

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
