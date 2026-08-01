"""Shared fixtures.

The processor is built from a vocab written to a temp dir rather than pulled
from the Hub, so the core tests need no network and run in well under a second.
"""

from __future__ import annotations

import json
import string

import pytest
from transformers import Wav2Vec2CTCTokenizer, Wav2Vec2FeatureExtractor, Wav2Vec2Processor


@pytest.fixture(scope="session")
def vocab() -> dict[str, int]:
    """wav2vec2's CTC vocabulary: blank first, then the delimiter and characters."""
    tokens = ["<pad>", "<s>", "</s>", "<unk>", "|", *string.ascii_uppercase, "'"]
    return {token: index for index, token in enumerate(tokens)}


@pytest.fixture(scope="session")
def processor(tmp_path_factory, vocab) -> Wav2Vec2Processor:
    vocab_path = tmp_path_factory.mktemp("vocab") / "vocab.json"
    vocab_path.write_text(json.dumps(vocab))

    tokenizer = Wav2Vec2CTCTokenizer(
        str(vocab_path),
        unk_token="<unk>",
        pad_token="<pad>",
        word_delimiter_token="|",
    )
    feature_extractor = Wav2Vec2FeatureExtractor(
        feature_size=1,
        sampling_rate=16_000,
        padding_value=0.0,
        do_normalize=True,
        return_attention_mask=True,
    )
    return Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)
