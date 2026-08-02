"""The fixed CTC vocabulary shared by every checkpoint this repo trains."""

from __future__ import annotations

import json

from asr.vocab import CTC_VOCAB, write_vocab_json


def test_pad_is_token_zero():
    """CTCLoss assumes blank==0; this is what makes pad_token_id double as it."""
    assert CTC_VOCAB["<pad>"] == 0


def test_vocab_covers_the_normalized_label_space():
    import string

    for letter in string.ascii_uppercase:
        assert letter in CTC_VOCAB
    assert "'" in CTC_VOCAB
    assert "|" in CTC_VOCAB  # word delimiter


def test_ids_are_unique_and_dense():
    ids = sorted(CTC_VOCAB.values())
    assert ids == list(range(len(CTC_VOCAB)))


def test_write_vocab_json_round_trips(tmp_path):
    path = write_vocab_json(tmp_path / "vocab.json")
    assert json.loads(path.read_text()) == CTC_VOCAB
