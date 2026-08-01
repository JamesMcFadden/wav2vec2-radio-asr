"""Audio decoding and dataset loading.

Everything that touches the Hub is marked ``network`` -- run the fast subset
with ``uv run pytest -m 'not network'``.
"""

from __future__ import annotations

import io

import numpy as np
import pytest
import soundfile as sf

from asr.data import TARGET_SAMPLE_RATE, decode_audio, load_librispeech


def _flac_bytes(audio: np.ndarray, sample_rate: int) -> bytes:
    buffer = io.BytesIO()
    sf.write(buffer, audio, sample_rate, format="FLAC")
    return buffer.getvalue()


def test_decode_audio_reads_flac_bytes():
    audio = np.sin(np.linspace(0, 100, 16_000)).astype("float32")
    decoded = decode_audio({"bytes": _flac_bytes(audio, 16_000), "path": "x.flac"})

    assert decoded.dtype == np.float32
    assert decoded.shape == (16_000,)
    np.testing.assert_allclose(decoded, audio, atol=1e-4)


def test_decode_audio_resamples_to_16k():
    audio = np.sin(np.linspace(0, 100, 8_000)).astype("float32")
    decoded = decode_audio({"bytes": _flac_bytes(audio, 8_000), "path": "x.flac"})

    assert decoded.shape[0] == pytest.approx(16_000, rel=0.01)


def test_decode_audio_downmixes_to_mono():
    stereo = np.stack([np.ones(1_000), np.zeros(1_000)], axis=1).astype("float32")
    decoded = decode_audio({"bytes": _flac_bytes(stereo, 16_000), "path": "x.flac"})

    assert decoded.ndim == 1
    np.testing.assert_allclose(decoded, 0.5, atol=1e-4)


def test_decode_audio_without_bytes_or_path_raises():
    with pytest.raises(ValueError, match="neither 'bytes' nor"):
        decode_audio({})


def test_unknown_split_rejected_before_download():
    """Fail fast on a bad split rather than after a multi-GB download."""
    with pytest.raises(ValueError, match="is not published for config"):
        load_librispeech("clean", "test.clean")  # valid only for the "all" config


@pytest.mark.network
def test_dummy_corpus_loads_and_decodes():
    dataset = load_librispeech(dummy=True, max_samples=2)
    assert len(dataset) == 2

    audio = decode_audio(dataset[0]["audio"])
    assert audio.ndim == 1
    assert audio.dtype == np.float32
    assert len(audio) / TARGET_SAMPLE_RATE > 1.0  # a real utterance, seconds long
    assert dataset[0]["text"].isupper()
