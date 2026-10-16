import json, subprocess, sys, wave
import numpy as np
import pytest
from meetassist import export, store, paths

ROWS = [
    {"seq": 1, "track": "mic", "t_start": 1.0, "t_end": 3.0, "text": "Hello team."},
    {"seq": 2, "track": "loop", "t_start": 4.0, "t_end": 6.0, "text": "Hi, can you hear me?"},
    {"seq": 3, "track": "mic", "t_start": 7.0, "t_end": 8.0, "text": "Yes."},
    {"seq": 4, "track": "mic", "t_start": 8.5, "t_end": 9.5, "text": "Let's start."},
    {"seq": 5, "track": "mic", "t_start": 3661.5, "t_end": 3663, "text": "Later."},
]

def test_hms_and_srt_timestamps():
    assert export.hms(61) == "00:01:01" and export.hms(3661.5) == "01:01:01"
    assert export.srt_ts(65.5) == "00:01:05,500"

def test_grouping_merges_same_speaker_only():
    g = export.group_rows(ROWS)
    assert [x["track"] for x in g] == ["mic", "loop", "mic", "mic"]
    assert g[2]["text"] == "Yes. Let's start."

def test_txt_srt_md_and_cleaned_variant():
    txt = export.render_txt(ROWS)
    assert "[00:00:01] [You] Hello team." in txt and "[00:00:04] [Remote] Hi, can you hear me?" in txt
    assert txt.index("Hello team") < txt.index("Hi, can")
    srt = export.render_srt(ROWS)
    assert srt.startswith("1\n00:00:01,000 --> 00:00:03,000\nYou: Hello team.")
    cleaned = {1: "Hello, team!"}
    assert "Hello, team!" in export.render_txt(ROWS, "cleaned", cleaned) and "Hello team." in export.render_txt(ROWS)   # raw untouched
    md = export.render_md(ROWS, {"title": "T", "started_at": "now", "model": "small"}, summary="## Summary\n- x")
    assert md.startswith("# T") and "## Summary" in md and "## Transcript" in md

def test_journal_survives_torn_last_line(tmp_path):
    p = tmp_path / "journal.jsonl"
    p.write_text(json.dumps(ROWS[0]) + "\n" + json.dumps(ROWS[1]) + "\n" + '{"seq": 3, "track": "mi', encoding="utf-8")
    assert [r["seq"] for r in export.read_journal(p)] == [1, 2]

def test_store_roundtrip_and_decode(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "MEETINGS_DIR", tmp_path / "m")
    d = store.new_meeting("My meeting", "live", "small")
    assert store.read_meta(d)["title"] == "My meeting" and store.list_meetings()[0].path == d
    wav = tmp_path / "a.wav"
    with wave.open(str(wav), "wb") as w:
        w.setnchannels(2); w.setsampwidth(2); w.setframerate(44100)
        w.writeframes((np.sin(np.arange(44100 * 2) * 0.05) * 8000).astype(np.int16).repeat(2).tobytes())
    secs = store.decode_to_pcm(str(wav), d / "raw" / "mic.pcm")
    assert abs(secs - 2.0) < 0.05 and abs(store.audio_seconds(d) - 2.0) < 0.05
    store.delete_meeting(d); assert not d.exists()

def _run_worker(tmp_path, monkeypatch, language, seconds=25):
    sample = paths.APP_ROOT / "audio" / "Recording.m4a"
    if not sample.exists():
        pytest.skip("sample missing")
    monkeypatch.setattr(paths, "MEETINGS_DIR", tmp_path / "m")
    d = store.new_meeting("w", "import", "small")
    store.decode_to_pcm(str(sample), d / "raw" / "mic.pcm")
    pcm = d / "raw" / "mic.pcm"; pcm.write_bytes(pcm.read_bytes()[: 16000 * 2 * seconds])
    (d / "raw" / "END").write_text("x")
    r = subprocess.run([sys.executable, "-m", "meetassist.asr_worker", "--meeting-dir", str(d), "--model", "small",
                        "--language", language], cwd=str(paths.APP_ROOT), timeout=400, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-500:]
    w = json.loads((d / "worker.json").read_text())
    assert w["state"] == "done" and w["pieces"] >= 1
    rows = export.read_journal(d / "journal.jsonl")
    assert rows and all(0 <= x["t_start"] <= x["t_end"] <= seconds + 1 for x in rows)
    return rows


@pytest.mark.skipif(not (paths.MODELS_DIR / "ct2-small" / "model.bin").exists(), reason="small model not installed")
def test_worker_whisper_mode_end_to_end(tmp_path, monkeypatch):
    rows = _run_worker(tmp_path, monkeypatch, "whisper")
    assert "vanakkam" in " ".join(x["text"] for x in rows).lower()


@pytest.mark.skipif(not ((paths.MODELS_DIR / "ct2-small" / "model.bin").exists() and (paths.MODELS_DIR / "indic-asr" / "assets" / "encoder.onnx").exists()),
                    reason="models not installed")
def test_worker_hybrid_mode_gives_tamil_script_for_tamil_speech(tmp_path, monkeypatch):
    rows = _run_worker(tmp_path, monkeypatch, "hybrid")
    from meetassist import tamil
    text = " ".join(x["text"] for x in rows)
    assert tamil.has_tamil(text) and "நம்ம" in text                        # Tamil speech -> Tamil script, not English gibberish
    assert any(x.get("alt") for x in rows)                                   # the Whisper reading is kept as a hint


def test_echo_suppression_hides_mic_copy_of_remote_only():
    rows = [
        {"seq": 1, "track": "loop", "t_start": 10.0, "t_end": 14.0, "text": "Namma project deadline next week la irukku, please update status"},
        {"seq": 2, "track": "mic", "t_start": 10.4, "t_end": 14.2, "text": "namma project deadline next week la irukku please update the status"},   # speaker bleed
        {"seq": 3, "track": "mic", "t_start": 15.0, "t_end": 17.0, "text": "Sure, I will share it by tomorrow"},                                     # genuinely you
        {"seq": 4, "track": "mic", "t_start": 60.0, "t_end": 62.0, "text": "namma project deadline next week"},                                          # same words but far away in time
    ]
    kept = [r["seq"] for r in export.drop_echo(rows)]
    assert kept == [1, 3, 4]
    assert export.drop_echo([r for r in rows if r["track"] == "mic"]) == [r for r in rows if r["track"] == "mic"]   # no loop track -> untouched


def test_reader_hides_rows_flagged_as_echo_but_journal_keeps_them(tmp_path):
    p = tmp_path / "journal.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in [
        {"seq": 1, "track": "mic", "t_start": 1, "t_end": 2, "text": "real me"},
        {"seq": 2, "track": "mic", "t_start": 3, "t_end": 4, "text": "bleed", "echo": True}]), encoding="utf-8")
    assert [r["seq"] for r in export.read_journal(p)] == [1]
    assert [r["seq"] for r in export.read_journal(p, hide_echo=False)] == [1, 2]


def test_display_modes_english_roman_tamil():
    rows = [{"seq": 1, "track": "mic", "t_start": 1, "t_end": 3, "text": "இது ஒரு டிரிப்புக்காக", "lang": "ta", "en": "This is for a trip", "alt": "idhu oru"},
            {"seq": 2, "track": "mic", "t_start": 4, "t_end": 5, "text": "Sounds good.", "lang": "en"},
            {"seq": 3, "track": "mic", "t_start": 6, "t_end": 7, "text": "நம்ம போலாம்", "lang": "ta"}]            # translation missing -> romanised fallback
    eng = export.render_txt(rows, roman="english")
    assert "This is for a trip Sounds good." in eng and "இ" not in eng and "namma" in eng
    assert "இது ஒரு" in export.render_txt(rows, roman="tamil")
    assert "idhu oru" in export.render_txt(rows, roman="roman").lower() or "idhu" in export.render_txt(rows, roman="roman")
