"""AI4Bharat IndicConformer-600M (multilingual, CTC head) running on onnxruntime — no torch needed.

The published model ships its mel front-end as a TorchScript file; we re-implement NeMo's standard
80-bin log-mel front-end in numpy (16 kHz, 25 ms Hann / 10 ms hop, n_fft 512, pre-emphasis 0.97,
slaney mel, log guard 2^-24, per-feature normalisation) and decode with greedy CTC restricted to the
language's own token subset.  Output is in the language's native script (Tamil script for 'ta').
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import List

import numpy as np

SR, N_FFT, WIN, HOP, N_MELS = 16000, 512, 400, 160, 80
BLANK = 256


def _hz_to_mel(f):
    f = np.asarray(f, dtype=np.float64)
    mel = f / (200.0 / 3)
    log_t = np.log(6.4) / 27.0
    return np.where(f >= 1000.0, 15.0 + np.log(np.maximum(f, 1e-9) / 1000.0) / log_t, mel)


def _mel_to_hz(m):
    m = np.asarray(m, dtype=np.float64)
    hz = m * (200.0 / 3)
    log_t = np.log(6.4) / 27.0
    return np.where(m >= 15.0, 1000.0 * np.exp(log_t * (m - 15.0)), hz)


def mel_filterbank(sr=SR, n_fft=N_FFT, n_mels=N_MELS) -> np.ndarray:
    fft_f = np.linspace(0, sr / 2, 1 + n_fft // 2)
    mel_f = _mel_to_hz(np.linspace(_hz_to_mel(0.0), _hz_to_mel(sr / 2), n_mels + 2))
    fdiff = np.diff(mel_f)
    ramps = np.subtract.outer(mel_f, fft_f)
    lower = -ramps[:-2] / fdiff[:-1, None]
    upper = ramps[2:] / fdiff[1:, None]
    w = np.maximum(0, np.minimum(lower, upper))
    w *= (2.0 / (mel_f[2:n_mels + 2] - mel_f[:n_mels]))[:, None]          # slaney area normalisation
    return w.astype(np.float32)


_FB = None
_WIN = None


def features(audio: np.ndarray) -> np.ndarray:
    """float32 mono 16 kHz -> (1, 80, T) normalised log-mel."""
    global _FB, _WIN
    if _FB is None:
        _FB = mel_filterbank()
        w = np.zeros(N_FFT, np.float32)
        o = (N_FFT - WIN) // 2
        w[o:o + WIN] = np.hanning(WIN).astype(np.float32)                   # torch.hann_window(periodic=False)
        _WIN = w
    x = audio.astype(np.float32)
    x = np.concatenate([x[:1], x[1:] - 0.97 * x[:-1]])
    n_frames = len(x) // HOP + 1
    xp = np.pad(x, N_FFT // 2, mode="reflect")
    idx = np.arange(N_FFT)[None, :] + HOP * np.arange(n_frames)[:, None]
    spec = np.fft.rfft(xp[idx] * _WIN[None, :], axis=1)
    power = (spec.real ** 2 + spec.imag ** 2).astype(np.float32)           # (T, 257)
    mel = np.log(power @ _FB.T + 2.0 ** -24)                               # (T, 80)
    mean = mel.mean(axis=0, keepdims=True)
    std = mel.std(axis=0, ddof=1, keepdims=True) if n_frames > 1 else np.ones_like(mean)
    mel = (mel - mean) / (std + 1e-5)
    return mel.T[None].astype(np.float32)


def _log_softmax(x: np.ndarray) -> np.ndarray:
    x = x - x.max(axis=-1, keepdims=True)
    return x - np.log(np.exp(x).sum(axis=-1, keepdims=True))


class IndicConformer:
    def __init__(self, model_dir: Path, threads: int = 4) -> None:
        import onnxruntime as ort
        a = Path(model_dir) / "assets"
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = threads
        opts.log_severity_level = 3
        prov = ["CPUExecutionProvider"]
        self.enc = ort.InferenceSession(str(a / "encoder.onnx"), sess_options=opts, providers=prov)
        self.ctc = ort.InferenceSession(str(a / "ctc_decoder.onnx"), sess_options=opts, providers=prov)
        self.vocab = json.loads((a / "vocab.json").read_text(encoding="utf-8"))
        self.masks = json.loads((a / "language_masks.json").read_text(encoding="utf-8"))

    def transcribe(self, audio: np.ndarray, lang: str = "ta") -> str:
        return self.transcribe_ex(audio, lang)[0]

    def transcribe_ex(self, audio: np.ndarray, lang: str = "ta"):
        """Returns (text, confidence): confidence = mean best-token probability over emitted tokens (0..1);
        near-silent / noise-only input decodes to low-confidence gibberish."""
        feats = features(audio)
        enc, enc_len = self.enc.run(["outputs", "encoded_lengths"],
                                    {"audio_signal": feats, "length": np.array([feats.shape[2]], dtype=np.int64)})
        logp = self.ctc.run(["logprobs"], {"encoder_output": enc})[0]       # (1, T', 5633)
        logp = _log_softmax(logp[:, :, self.masks[lang]])[0]
        ids = logp.argmax(axis=-1)
        best = np.exp(logp.max(axis=-1))
        out: List[str] = []
        confs: List[float] = []
        prev = -1
        v = self.vocab[lang]
        for t, i in enumerate(ids):
            if i != prev and i != BLANK:
                out.append(v[i]); confs.append(float(best[t]))
            prev = i
        text = "".join(out).replace("▁", " ").strip()
        return text, (float(np.mean(confs)) if confs else 0.0)
