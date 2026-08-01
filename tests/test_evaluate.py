"""Batching and device selection for the evaluation entry point."""

from __future__ import annotations

import io

import numpy as np
import soundfile as sf

from asr.evaluate import iter_batches, pick_device


def label_for(n_samples: int) -> str:
    """Encode a length as letters -- ``normalize_text`` strips digits, by design."""
    return "utterance " + "X" * (n_samples // 1000)


class FakeDataset:
    """Minimal stand-in for a ``datasets.Dataset``: ``len()`` plus slice-to-dict."""

    def __init__(self, lengths: list[int]):
        self.lengths = lengths

    def __len__(self) -> int:
        return len(self.lengths)

    def __getitem__(self, key: slice) -> dict[str, list]:
        chosen = self.lengths[key]
        audio = []
        for n in chosen:
            buffer = io.BytesIO()
            sf.write(buffer, np.zeros(n, dtype="float32"), 16_000, format="FLAC")
            audio.append({"bytes": buffer.getvalue(), "path": f"{n}.flac"})
        return {"audio": audio, "text": [label_for(n) for n in chosen]}


def _collect(dataset, **kwargs):
    return list(iter_batches(dataset, **kwargs))


def test_every_utterance_is_emitted_exactly_once():
    lengths = [1000, 5000, 2000, 8000, 3000, 100, 400]
    batches = _collect(FakeDataset(lengths), batch_size=3, sort_by_length=False)

    emitted = [len(audio) for audio, _ in batches for audio in audio]
    assert sorted(emitted) == sorted(lengths)
    assert sum(len(texts) for _, texts in batches) == len(lengths)


def test_batches_respect_batch_size():
    batches = _collect(FakeDataset([1000] * 7), batch_size=3, sort_by_length=False)
    assert [len(a) for a, _ in batches] == [3, 3, 1]


def test_unsorted_preserves_dataset_order():
    lengths = [5000, 1000, 3000]
    batches = _collect(FakeDataset(lengths), batch_size=3, sort_by_length=False)
    assert [len(a) for a in batches[0][0]] == lengths


def test_sorting_groups_similar_lengths_within_the_buffer():
    lengths = [8000, 1000, 7000, 2000]
    batches = _collect(FakeDataset(lengths), batch_size=2, sort_by_length=True, buffer_batches=2)
    # Buffer covers all four, so the two short clips batch together, then the long pair.
    assert [len(a) for a in batches[0][0]] == [1000, 2000]
    assert [len(a) for a in batches[1][0]] == [7000, 8000]


def test_sorting_is_bounded_by_the_buffer():
    """A long clip beyond the buffer must not migrate into an earlier batch."""
    lengths = [5000, 6000, 1000, 1100]
    batches = _collect(FakeDataset(lengths), batch_size=2, sort_by_length=True, buffer_batches=1)
    assert [len(a) for a in batches[0][0]] == [5000, 6000]
    assert [len(a) for a in batches[1][0]] == [1000, 1100]


def test_audio_and_text_stay_aligned_through_sorting():
    lengths = [8000, 1000]
    ((audio, texts),) = _collect(
        FakeDataset(lengths), batch_size=2, sort_by_length=True, buffer_batches=1
    )
    for waveform, text in zip(audio, texts, strict=True):
        assert text == label_for(len(waveform)).upper()


def test_pick_device_honours_explicit_request():
    assert pick_device("cpu").type == "cpu"


def test_pick_device_auto_returns_something_usable():
    assert pick_device("auto").type in {"cpu", "mps", "cuda"}
