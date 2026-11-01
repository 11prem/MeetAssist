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


def test_resume_offsets_are_max_committed_end_per_track(tmp_path):
    from meetassist.asr_worker import resume_offsets
    p = tmp_path / "journal.jsonl"
    assert resume_offsets(p) == {"mic": 0.0, "loop": 0.0}                  # fresh run
    p.write_text("\n".join(json.dumps(r) for r in ROWS[:4]) + "\n"
                 + json.dumps({"seq": 6, "track": "mic", "t_start": 9.6, "t_end": 11.0, "text": "x", "echo": True})
                 + '\n{"seq": 7, "track": "loop", "t_end": 99', encoding="utf-8")   # echo rows count, torn line ignored
    assert resume_offsets(p) == {"mic": 11.0, "loop": 6.0}


@pytest.mark.skipif(not (paths.MODELS_DIR / "ct2-small" / "model.bin").exists(), reason="small model not installed")
def test_respawned_worker_resumes_after_last_committed_row(tmp_path, monkeypatch):
    sample = paths.APP_ROOT / "audio" / "Recording.m4a"
    if not sample.exists():
        pytest.skip("sample missing")
    monkeypatch.setattr(paths, "MEETINGS_DIR", tmp_path / "m")
    d = store.new_meeting("w", "import", "small")
    store.decode_to_pcm(str(sample), d / "raw" / "mic.pcm")
    pcm = d / "raw" / "mic.pcm"; pcm.write_bytes(pcm.read_bytes()[: 16000 * 2 * 25])
    (d / "raw" / "END").write_text("x")
    (d / "journal.jsonl").write_text(json.dumps({"seq": 1, "track": "mic", "t_start": 0.5, "t_end": 12.0, "text": "earlier"}) + "\n", encoding="utf-8")
    r = subprocess.run([sys.executable, "-m", "meetassist.asr_worker", "--meeting-dir", str(d), "--model", "small",
                        "--language", "whisper"], cwd=str(paths.APP_ROOT), timeout=400, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-500:]
    rows = export.read_journal(d / "journal.jsonl")
    assert rows[0]["text"] == "earlier" and len(rows) > 1
    assert all(x["t_start"] >= 12.0 - 0.01 and x["seq"] > 1 for x in rows[1:]), "a resumed worker must not redo committed audio"


def test_journal_append_after_torn_last_line_starts_a_new_line(tmp_path):
    from meetassist.asr_worker import Journal
    from meetassist.transcriber import Piece
    p = tmp_path / "journal.jsonl"
    p.write_text(json.dumps(ROWS[0]) + "\n" + '{"seq": 2, "track": "mi', encoding="utf-8")
    j = Journal(p); j.add(Piece("mic", 5.0, 6.0, "after the crash", "en"), "small"); j.f.close()
    assert [r["text"] for r in export.read_journal(p)] == ["Hello team.", "after the crash"]
    assert '{"seq": 2, "track": "mi\n' in p.read_text(encoding="utf-8")       # the torn text itself is left as it was


def test_delete_with_an_open_memmap_keeps_the_meeting_listed_and_retryable(tmp_path, monkeypatch):
    """An open handle on raw audio (voice ID's memmap) must not leave a hidden folder still holding the pcm."""
    monkeypatch.setattr(paths, "MEETINGS_DIR", tmp_path / "m")
    d = store.new_meeting("busy")
    (d / "raw" / "mic.pcm").write_bytes(np.ones(1000, np.int16).tobytes()); (d / "journal.jsonl").write_text("{}\n")
    held = [np.memmap(d / "raw" / "mic.pcm", dtype=np.int16, mode="r")]
    assert store.delete_meeting(d) is False
    assert [m.path for m in store.list_meetings()] == [d]
    held.clear()                                              # memmap closed
    assert store.delete_meeting(d) is True and not d.exists()
