"""User settings (JSON in %LOCALAPPDATA%\\MeetAssist). The API key is DPAPI-encrypted for the
current Windows user, so it is never stored in plain text (and is not portable between laptops —
by design: enter it once per machine)."""
from __future__ import annotations

import base64
import ctypes
import json
from ctypes import wintypes
from dataclasses import asdict, dataclass, fields

from . import paths


class _Blob(ctypes.Structure):
    _fields_ = [("cbData", wintypes.DWORD), ("pbData", ctypes.POINTER(ctypes.c_char))]


def _blob(b: bytes):
    buf = ctypes.create_string_buffer(b, len(b))
    return _Blob(len(b), ctypes.cast(buf, ctypes.POINTER(ctypes.c_char))), buf


def protect(text: str) -> str:
    if not text:
        return ""
    try:
        inb, _k = _blob(text.encode("utf-8")); out = _Blob()
        if not ctypes.windll.crypt32.CryptProtectData(ctypes.byref(inb), None, None, None, None, 0, ctypes.byref(out)):
            raise OSError("CryptProtectData failed")
        data = ctypes.string_at(out.pbData, out.cbData); ctypes.windll.kernel32.LocalFree(out.pbData)
        return "dpapi:" + base64.b64encode(data).decode()
    except Exception:
        return "b64:" + base64.b64encode(text.encode()).decode()      # last resort, still not plain


def unprotect(s: str) -> str:
    if not s:
        return ""
    try:
        if s.startswith("b64:"):
            return base64.b64decode(s[4:]).decode()
        inb, _k = _blob(base64.b64decode(s[6:])); out = _Blob()
        if not ctypes.windll.crypt32.CryptUnprotectData(ctypes.byref(inb), None, None, None, None, 0, ctypes.byref(out)):
            return ""
        data = ctypes.string_at(out.pbData, out.cbData); ctypes.windll.kernel32.LocalFree(out.pbData)
        return data.decode("utf-8")
    except Exception:
        return ""


@dataclass
class Settings:
    llm_base_url: str = ""
    llm_api_key: str = ""            # plaintext in memory only
    llm_model: str = "deepseek-chat"
    llm_verify_tls: bool = True
    auto_postprocess: bool = True    # run LLM cleanup automatically after a meeting (summary is a manual button)
    language: str = "hybrid"         # hybrid | en | ta | whisper  (see asr_worker)
    tamil_display: str = "english"   # Tamil speech shown as: "english" (translation, default) | "roman" (Tamil in English letters) | "tamil" script
    version: int = 3                 # settings schema version (v3: English translation is the default display)
    model_name: str = "small"
    refine_model: str = ""           # "" = off; else re-transcribe in the background after the meeting with this model
    vocabulary: str = ""             # names / terms to bias recognition, e.g. "Python, LLM, Kubernetes"
    mic_name: str = ""               # substring of preferred mic device; "" = Windows default
    record_system_audio: bool = True
    appearance: str = "system"       # system | light | dark
    ack_notice: bool = False         # one-time recording-consent reminder shown
    record_mic: bool = True
    voice_enroll_asked: bool = False  # one-time "record my voice" offer shown (Settings > Voices repeats it on request)

    @staticmethod
    def load() -> "Settings":
        s = Settings()
        try:
            d = json.loads(paths.SETTINGS_FILE.read_text(encoding="utf-8"))
            for f in fields(Settings):
                if f.name in d and f.name != "llm_api_key":
                    setattr(s, f.name, d[f.name])
            s.llm_api_key = unprotect(d.get("llm_api_key_enc", ""))
            if int(d.get("version", 1)) < 3:
                s.tamil_display = "english"                 # user preference: Tamil speech must come out as English
                s.version = 3
            if s.language not in ("hybrid", "en", "ta", "whisper"):
                s.language = "hybrid"                       # old values ('mixed', 'auto') -> the new default
        except Exception:
            pass
        return s

    def save(self) -> None:
        paths.DATA_DIR.mkdir(parents=True, exist_ok=True)
        d = asdict(self)
        d.pop("llm_api_key", None)
        d["llm_api_key_enc"] = protect(self.llm_api_key)
        paths.SETTINGS_FILE.write_text(json.dumps(d, indent=2), encoding="utf-8")
