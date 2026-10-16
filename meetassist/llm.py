"""Company LLM client (OpenAI-compatible /chat/completions). The ONLY network use in the app.

stdlib only (urllib) — no extra dependency. The API key is never logged.
All post-processing is *additive*: raw transcript text is never modified (AGENTS rule 4);
cleaned / summarised text is stored in separate files by the caller.
"""
from __future__ import annotations

import json
import re
import ssl
import time
import urllib.error
import urllib.request
from typing import Callable, List, Optional, Sequence

CLEAN_SYSTEM = (
    "You are a meticulous transcript editor for business meetings held in Tamil, English and Tanglish. "
    "You receive numbered transcript lines produced by speech recognition, which contains mistakes. "
    "Lines are either English (Latin letters) or Tamil written in Tamil script, where English words and "
    "technical terms are often spelled phonetically in Tamil script. Some lines carry a hint after '|| alt:' - "
    "a second recogniser's reading of the same audio, often spelling English terms and names correctly; use it as "
    "evidence, do not copy it blindly, and never output the hint marker. "
    "For every line: fix recognition errors, Tamil spelling, punctuation; remove stutters and fillers; "
    "write English words and technical terms (e.g. 'மெஷின் லேர்னிங்', 'யூடியூப்') in English letters "
    "('machine learning', 'YouTube') while keeping the Tamil words in Tamil script, as a Tamil professional would type. "
    "STRICT RULES: keep each speaker's meaning and language mix; do NOT summarise, shorten, merge, split, reorder, "
    "add content or translate whole sentences; if a line is unintelligible keep it unchanged. "
    "Reply with ONLY a JSON array of strings, the same length and order as the input."
)

CLEAN_ENGLISH_SYSTEM = (
    "You are a meticulous editor of meeting transcripts. Each numbered line is an ENGLISH rendering of what a speaker said "
    "(speakers mix English and Tamil; Tamil parts were machine-translated into English). Some lines carry a hint after "
    "'|| alt:' - the original Tamil words (in Tamil script) from a Tamil speech recogniser. Use the hint to repair "
    "mistranslations, wrong names and technical terms, then write natural, fluent, correct English with proper punctuation "
    "(for example 'athu ethukaaga na its for a trip' -> 'that is for a trip'). Keep the speaker's meaning exactly; do NOT "
    "summarise, shorten, merge, split, reorder or add content. Never output the hint marker. "
    "Reply with ONLY a JSON array of strings, the same length and order as the input."
)

CLEAN_ROMAN_ADDENDUM = (
    " OUTPUT STYLE: write the Tamil words in English letters too (natural Tanglish as people type in chat, "
    "e.g. 'ippo namma meeting la irukkom'), so the whole line is in Latin letters."
)

SUMMARY_SYSTEM = (
    "You write concise, accurate meeting notes from a transcript that may mix Tamil, English and "
    "Tanglish. Write the notes in English. Never invent facts that are not in the transcript. "
    "Use exactly these markdown sections: '## Summary' (5-8 bullets), '## Key decisions', "
    "'## Action items' (a checklist: '- [ ] task — owner (if mentioned) — due (if mentioned)'), "
    "'## Open questions'. Write 'None' for an empty section."
)

TRANSLATE_SYSTEM = (
    "Translate the numbered meeting-transcript lines (Tamil / Tanglish / English mix) into clear "
    "natural English. Keep technical terms and names. Reply with ONLY a JSON array of strings, "
    "same length and order as the input."
)

ASK_SYSTEM = (
    "You answer questions about a single business meeting, using ONLY the transcript and summary given below. "
    "If the answer is not in the material, say so plainly instead of guessing. Be concise and specific "
    "(name the person, decision or number when the material has it)."
)


class LLMError(RuntimeError):
    pass


def _endpoint(base: str) -> str:
    b = base.strip().rstrip("/")
    if b.endswith("/chat/completions"):
        return b
    if re.search(r"/v\d+$", b):
        return b + "/chat/completions"
    return b + "/v1/chat/completions"


class LLMClient:
    def __init__(self, base_url: str, api_key: str, model: str = "deepseek-chat",
                 verify_tls: bool = True, timeout: float = 120.0, retries: int = 3) -> None:
        self.base_url, self.api_key, self.model = base_url, api_key, model
        self.timeout, self.retries = timeout, retries
        self._ctx = ssl.create_default_context()
        if not verify_tls:
            self._ctx.check_hostname = False
            self._ctx.verify_mode = ssl.CERT_NONE

    @property
    def available(self) -> bool:
        return bool(self.base_url.strip() and self.api_key.strip())

    def chat(self, system: str, user: str, temperature: float = 0.1, max_tokens: int = 4096) -> str:
        return self._request([{"role": "system", "content": system}, {"role": "user", "content": user}], temperature, max_tokens)

    def ask(self, context: str, history: Sequence[dict], question: str, temperature: float = 0.2, max_tokens: int = 1024) -> str:
        """Q&A about one meeting. history: prior [{'role': 'user'|'assistant', 'content': str}, ...] turns."""
        # shortcut: context is sent in full every time, no chunking; a very long (multi-hour) meeting could exceed
        # the model's context window - chunk like summarize() if that happens in practice.
        messages = [{"role": "system", "content": ASK_SYSTEM + "\n\n--- Meeting content ---\n" + context}]
        messages.extend(history)
        messages.append({"role": "user", "content": question})
        return self._request(messages, temperature, max_tokens)

    def _request(self, messages: list, temperature: float, max_tokens: int) -> str:
        if not self.available:
            raise LLMError("LLM base URL / API key not configured (Settings).")
        body = json.dumps({"model": self.model, "temperature": temperature, "max_tokens": max_tokens, "messages": messages}).encode("utf-8")
        req = urllib.request.Request(_endpoint(self.base_url), data=body, method="POST", headers={
            "Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"})
        last: Optional[Exception] = None
        for attempt in range(self.retries):
            try:
                with urllib.request.urlopen(req, timeout=self.timeout, context=self._ctx) as r:
                    data = json.loads(r.read().decode("utf-8"))
                return (data["choices"][0]["message"]["content"] or "").strip()
            except urllib.error.HTTPError as e:                   # auth/4xx: retrying will not help
                detail = e.read().decode("utf-8", "replace")[:300]
                if e.code in (401, 403, 404, 400):
                    raise LLMError(f"HTTP {e.code}: {detail}") from None
                last = LLMError(f"HTTP {e.code}: {detail}")
            except Exception as e:                                 # network / timeout / bad JSON
                last = e
            time.sleep(1.5 * (2 ** attempt))
        raise LLMError(f"LLM request failed: {last}")

    def ping(self) -> str:
        return self.chat("Reply with the single word: ok", "ping", max_tokens=8)

    # ------------------------------------------------------------------ high-level helpers
    def _map_lines(self, system: str, lines: Sequence[str], progress: Optional[Callable[[int, int], None]],
                   batch_chars: int = 5000, originals: Optional[Sequence[str]] = None) -> List[str]:
        out: List[str] = []
        orig = list(originals) if originals is not None else list(lines)
        batches, ob, cur, ocur, size = [], [], [], [], 0
        for ln, o in zip(lines, orig):
            if cur and size + len(ln) > batch_chars:
                batches.append(cur); ob.append(ocur); cur, ocur, size = [], [], 0
            cur.append(ln); ocur.append(o); size += len(ln) + 8
        if cur:
            batches.append(cur); ob.append(ocur)
        for bi, b in enumerate(batches):
            payload = "\n".join(f"{i + 1}. {t}" for i, t in enumerate(b))
            fixed = list(ob[bi])
            try:
                arr = _json_array(self.chat(system, payload, max_tokens=4096))
                if len(arr) == len(b) and all(isinstance(x, str) for x in arr):
                    fixed = [x.strip() or o for x, o in zip(arr, ob[bi])]
            except LLMError:
                raise
            except Exception:
                pass                                           # malformed reply: keep raw lines for this batch
            out.extend(fixed)
            if progress:
                progress(bi + 1, len(batches))
        return out

    def clean_lines(self, lines: Sequence[str], progress=None, alts: Optional[Sequence[str]] = None,
                    roman: bool = False, english: bool = False) -> List[str]:
        """alts[i] = other recogniser's reading of line i (hint, may be empty)."""
        payload = [f"{l} || alt: {alts[i]}" if (alts and alts[i]) else l for i, l in enumerate(lines)]
        system = CLEAN_ENGLISH_SYSTEM if english else CLEAN_SYSTEM + (CLEAN_ROMAN_ADDENDUM if roman else "")
        return self._map_lines(system, payload, progress,
                               originals=list(lines))

    def translate_lines(self, lines: Sequence[str], progress=None) -> List[str]:
        return self._map_lines(TRANSLATE_SYSTEM, lines, progress)

    def summarize(self, transcript: str, progress=None) -> str:
        """Map-reduce for long meetings: chunk -> partial notes -> merged notes."""
        chunks, cur = [], ""
        for ln in transcript.splitlines():
            if len(cur) + len(ln) > 14000 and cur:
                chunks.append(cur); cur = ""
            cur += ln + "\n"
        if cur.strip():
            chunks.append(cur)
        if not chunks:
            return "## Summary\n- (empty transcript)\n"
        if len(chunks) == 1:
            if progress: progress(1, 1)
            return self.chat(SUMMARY_SYSTEM, chunks[0], max_tokens=2048)
        parts = []
        for i, c in enumerate(chunks):
            parts.append(self.chat(SUMMARY_SYSTEM, f"(Part {i + 1} of {len(chunks)} of the meeting)\n" + c, max_tokens=1500))
            if progress: progress(i + 1, len(chunks) + 1)
        merged = self.chat(SUMMARY_SYSTEM + " You are merging partial notes of the same meeting: dedupe and combine.",
                           "\n\n".join(parts), max_tokens=2048)
        if progress: progress(len(chunks) + 1, len(chunks) + 1)
        return merged


def _json_array(text: str) -> list:
    t = text.strip()
    t = re.sub(r"^```(?:json)?\s*|\s*```$", "", t)
    a, b = t.find("["), t.rfind("]")
    return json.loads(t[a:b + 1])
