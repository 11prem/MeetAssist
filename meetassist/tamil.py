"""Tamil text helpers: clean ASR artefacts, detect Tamil-ness, romanise to readable 'Tanglish'."""
from __future__ import annotations

import re

_SIGNS = "ாிீுூெேைொோௌ்"      # vowel signs + pulli
_DUP = re.compile(f"([{_SIGNS}])\\1+")
_ORPHAN = re.compile(f"(?<![க-ஹ]){'[' + _SIGNS + ']'}+")                            # sign with no consonant before it
_TAMIL_CH = re.compile("[஀-௿]")


def clean_tamil(text: str) -> str:
    """IndicConformer sometimes emits a repeated vowel sign ('போோது') or a dangling sign; fix both."""
    t = _DUP.sub(r"\1", text)
    t = _ORPHAN.sub("", t)
    return re.sub(r"\s+", " ", t).strip()


# very common colloquial/standard Tamil words & particles that never occur in transliterated English
_MARKERS = ("நம்ம", "பண்ண", "இருக்க", "இருந்த", "ஆனா", "அது", "இது", "என்ன", "சரி", "இல்ல", "வேணும்", "போறோம்",
            "பாக்க", "பார்", "ரொம்ப", "எல்லா", "கொஞ்ச", "மாதிரி", "அப்படி", "இப்படி", "நான்", "நீங்க", "அவங்க", "இன்னும்",
            "வச்ச", "சொல்ல", "பேச", "தமிழ்", "ங்க", "ல்ல", "ன்னு", "ணும்", "கிட்ட", "பத்தி", "அதுக்கு", "இதுக்கு",
            "எடுத்து", "நடக்க", "தெரிய", "வந்த", "போன", "உங்க", "எங்க", "இங்க", "அங்க", "ஆமா", "இல்லை", "உள்ள", "எப்படி", "நமக்கு", "பண்றோ", "பிடிக்க", "வாங்க", "இருக்கோ", "அப்புறம்", "பாத்த", "சொன்ன", "போல", "வேற", "ஏன்", "எனக்கு", "உனக்கு", "அவன்", "இவன்")


def tamil_marker_count(text: str) -> int:
    return sum(text.count(m) for m in _MARKERS)


def has_tamil(text: str) -> bool:
    return bool(_TAMIL_CH.search(text))


# ----------------------------------------------------------------------------- romanisation
_VOW = {"அ": "a", "ஆ": "aa", "இ": "i", "ஈ": "ee", "உ": "u", "ஊ": "oo", "எ": "e", "ஏ": "e", "ஐ": "ai",
        "ஒ": "o", "ஓ": "o", "ஔ": "au"}
_SIGN = {"ா": "aa", "ி": "i", "ீ": "ee", "ு": "u", "ூ": "oo", "ெ": "e", "ே": "e", "ை": "ai",
         "ொ": "o", "ோ": "o", "ௌ": "au", "": "a"}
# consonant -> (word-initial/geminate, between vowels / after nasal)
_CON = {"க": ("k", "g"), "ங": ("ng", "ng"), "ச": ("s", "s"), "ஞ": ("nj", "nj"), "ட": ("t", "d"), "ண": ("n", "n"),
        "த": ("th", "dh"), "ந": ("n", "n"), "ப": ("p", "b"), "ம": ("m", "m"), "ய": ("y", "y"), "ர": ("r", "r"),
        "ல": ("l", "l"), "வ": ("v", "v"), "ழ": ("zh", "zh"), "ள": ("l", "l"), "ற": ("r", "r"), "ன": ("n", "n"),
        "ஜ": ("j", "j"), "ஷ": ("sh", "sh"), "ஸ": ("s", "s"), "ஹ": ("h", "h"), "": ("f", "f")}
_NASAL = set("ங ஞ ண ந ம ன".split())


def _word(w: str) -> str:
    w = w.replace("ஃப", "")                      # ஃப = f
    out: list[str] = []
    i, n = 0, len(w)
    prev_vowel = False                      # previous emitted unit ended in a vowel
    prev_con = None                         # previous consonant char (for gemination / nasal)
    while i < n:
        ch = w[i]
        if ch in _VOW:
            out.append(_VOW[ch]); prev_vowel, prev_con = True, None; i += 1
        elif ch in _CON:
            nxt = w[i + 1] if i + 1 < n else ""
            if ch == "க" and nxt == "்" and i + 2 < n and w[i + 2] == "ஷ":          # க்ஷ
                out.append("ksha"); i += 3; prev_vowel, prev_con = True, None; continue
            voiced = (prev_vowel or prev_con in _NASAL) and nxt != "்"
            base = _CON[ch][1 if voiced else 0]
            if prev_con == "ங" and ch == "க":              # ங்க = nga
                base = ""
            elif prev_con == "ஞ" and ch == "ச":            # ஞ்ச = nja
                base = ""
            if nxt == "்":
                out.append(base if not (prev_con == ch) else ""); prev_vowel = False; prev_con = ch; i += 2
            else:
                sign = _SIGN.get(nxt, None) if nxt in _SIGN and nxt else None
                vowel = sign if sign is not None else "a"
                out.append(base + vowel); prev_vowel, prev_con = True, None
                i += 2 if sign is not None else 1
        elif ch == "ஃ":
            out.append("h"); i += 1
        else:
            out.append(ch); prev_vowel, prev_con = False, None; i += 1
    s = "".join(out).replace("ss", "ch")
    return re.sub(r"(.)\1\1+", r"\1\1", s)


def romanize(text: str) -> str:
    """Tamil script -> readable Latin ('Tanglish-style'); non-Tamil text is left untouched."""
    return re.sub(r"[஀-௿]+", lambda m: _word(m.group(0)), text)
