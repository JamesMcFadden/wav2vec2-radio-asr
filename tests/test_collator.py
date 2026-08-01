"""The collator is where CTC pipelines break silently. Pin its behaviour down."""

from __future__ import annotations

import numpy as np
import pytest

from asr.collator import LABEL_PAD_ID, DataCollatorCTCWithPadding
from asr.data import normalize_text


@pytest.fixture
def collator(processor):
    return DataCollatorCTCWithPadding(processor=processor)


@pytest.fixture
def features(processor):
    """Two examples of deliberately different audio *and* label lengths."""
    texts = ["HELLO WORLD", "DON'T"]
    lengths = [16_000, 8_000]
    rng = np.random.default_rng(0)
    return [
        {
            "input_values": processor.feature_extractor(
                rng.standard_normal(n).astype("float32"), sampling_rate=16_000
            )["input_values"][0],
            "labels": processor.tokenizer(text)["input_ids"],
        }
        for text, n in zip(texts, lengths, strict=True)
    ]


def test_audio_pads_to_batch_max_with_mask(collator, features):
    batch = collator(features)

    assert batch["input_values"].shape == (2, 16_000)
    assert batch["attention_mask"].shape == (2, 16_000)
    # The shorter clip is masked off beyond its real length.
    assert batch["attention_mask"][0].sum() == 16_000
    assert batch["attention_mask"][1].sum() == 8_000


def test_labels_pad_with_ignore_sentinel_not_pad_token(collator, features, processor):
    batch = collator(features)
    labels = batch["labels"]

    lengths = [len(f["labels"]) for f in features]
    assert labels.shape == (2, max(lengths))

    # Real positions survive; padded positions become -100, never the pad id.
    assert (labels[1, : lengths[1]] != LABEL_PAD_ID).all()
    assert (labels[1, lengths[1] :] == LABEL_PAD_ID).all()
    assert (labels == processor.tokenizer.pad_token_id).sum() == 0


def test_labels_round_trip_back_to_source_text(collator, features, processor):
    from asr.metrics import decode_labels

    batch = collator(features)
    assert decode_labels(batch["labels"], processor) == ["HELLO WORLD", "DON'T"]


def test_pad_token_is_the_ctc_blank(processor):
    """CTCLoss assumes blank==0; a mismatch here trains a model that emits nothing."""
    assert processor.tokenizer.pad_token_id == 0


def test_normalize_text_preserves_apostrophes():
    assert normalize_text("don't") == "DON'T"
    assert normalize_text("HELLO,  WORLD!") == "HELLO WORLD"
    assert normalize_text("  spaced   out  ") == "SPACED OUT"
