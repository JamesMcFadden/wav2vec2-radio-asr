"""LibriSpeech loading, audio decoding, and transcript normalization.

Audio is decoded with ``soundfile`` rather than the ``datasets`` built-in
``Audio`` feature. As of ``datasets`` 5.x that feature delegates to
``torchcodec``, which dynamically links a *system* FFmpeg; libsndfile ships
inside the ``soundfile`` wheel and reads LibriSpeech's FLAC natively, so this
keeps the project free of non-Python dependencies.
"""

from __future__ import annotations

import io
import re
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
from datasets import Audio, Dataset, load_dataset

TARGET_SAMPLE_RATE = 16_000

LIBRISPEECH = "openslr/librispeech_asr"
LIBRISPEECH_DUMMY = "hf-internal-testing/librispeech_asr_dummy"

# Splits published for each config on the Hub. Note the naming is *not* uniform:
# the "clean"/"other" configs use bare split names, "all" uses dotted ones.
SPLITS: dict[str, tuple[str, ...]] = {
    "clean": ("train.100", "train.360", "validation", "test"),
    "other": ("train.500", "validation", "test"),
    "all": (
        "train.clean.100",
        "train.clean.360",
        "train.other.500",
        "validation.clean",
        "validation.other",
        "test.clean",
        "test.other",
    ),
}

# wav2vec2's CTC vocabulary is A-Z, apostrophe, and the word delimiter.
_DISALLOWED = re.compile(r"[^A-Z' ]+")


def normalize_text(text: str) -> str:
    """Fold a transcript into the CTC label space.

    LibriSpeech transcripts already arrive uppercase and unpunctuated, so this
    is mostly a guard against stray characters. Apostrophes are deliberately
    preserved: they are in the vocabulary, and stripping them rewrites DON'T as
    DONT, which inflates WER against a model that predicts the apostrophe.
    """
    return " ".join(_DISALLOWED.sub(" ", text.upper()).split())


def _resample(audio: np.ndarray, orig_sample_rate: int) -> np.ndarray:
    """Resample to 16 kHz. Imported lazily so decoding doesn't pull in torch."""
    import torch
    import torchaudio.functional as AF

    tensor = torch.from_numpy(audio).unsqueeze(0)
    resampled = AF.resample(tensor, orig_sample_rate, TARGET_SAMPLE_RATE)
    return resampled.squeeze(0).numpy()


def decode_audio(entry: dict[str, Any]) -> np.ndarray:
    """Decode one undecoded ``Audio`` entry to a mono float32 waveform at 16 kHz."""
    raw = entry.get("bytes")
    if raw is None:
        path = entry.get("path")
        if not path:
            raise ValueError("audio entry has neither 'bytes' nor a readable 'path'")
        raw = Path(path).read_bytes()

    audio, sample_rate = sf.read(io.BytesIO(raw), dtype="float32")
    if audio.ndim > 1:  # LibriSpeech is mono, but don't assume it of other corpora
        audio = audio.mean(axis=1)
    if sample_rate != TARGET_SAMPLE_RATE:
        audio = _resample(audio, sample_rate)
    return audio


def load_librispeech(
    config: str = "clean",
    split: str = "test",
    *,
    max_samples: int | None = None,
    dummy: bool = False,
) -> Dataset:
    """Load a LibriSpeech split with audio decoding deferred to :func:`decode_audio`.

    Set ``dummy=True`` for the 73-utterance smoke-test corpus (a few MB) instead
    of the real thing (test-clean alone is ~350 MB; the full train set is ~60 GB).
    """
    if dummy:
        dataset = load_dataset(LIBRISPEECH_DUMMY, split="validation")
    else:
        if config in SPLITS and split not in SPLITS[config]:
            raise ValueError(
                f"split {split!r} is not published for config {config!r}; "
                f"expected one of {SPLITS[config]}"
            )
        dataset = load_dataset(LIBRISPEECH, config, split=split)

    dataset = dataset.cast_column("audio", Audio(decode=False))
    if max_samples is not None:
        dataset = dataset.select(range(min(max_samples, len(dataset))))
    return dataset


def prepare_example(example: dict[str, Any], processor: Any) -> dict[str, Any]:
    """Turn a raw row into the ``input_values`` / ``labels`` pair CTC training wants.

    The feature extractor and tokenizer are addressed explicitly rather than
    through ``processor(...)``, whose routing between audio and text has changed
    across transformers releases.
    """
    audio = decode_audio(example["audio"])
    extracted = processor.feature_extractor(audio, sampling_rate=TARGET_SAMPLE_RATE)
    example["input_values"] = extracted["input_values"][0]
    example["labels"] = processor.tokenizer(normalize_text(example["text"]))["input_ids"]
    return example
