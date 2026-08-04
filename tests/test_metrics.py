"""WER arithmetic and greedy CTC decoding."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from asr.metrics import compute_wer, greedy_decode


def test_wer_is_zero_for_exact_match():
    assert compute_wer(["HELLO WORLD"], ["HELLO WORLD"]) == 0.0


def test_wer_counts_one_substitution_in_two_words():
    assert compute_wer(["HELLO THERE"], ["HELLO WORLD"]) == pytest.approx(0.5)


def test_wer_is_corpus_level_not_averaged():
    """4 errors over 6 reference words, not the mean of the per-utterance rates."""
    predictions = ["A B C", "X Y Z"]
    references = ["A B C", "P Q R"]
    assert compute_wer(predictions, references) == pytest.approx(3 / 6)


def test_empty_references_are_dropped():
    assert compute_wer(["ANYTHING", "HELLO"], ["", "HELLO"]) == 0.0


def test_all_empty_references_raises():
    with pytest.raises(ValueError, match="no non-empty references"):
        compute_wer(["A"], [""])


def test_length_mismatch_raises():
    with pytest.raises(ValueError, match="length mismatch"):
        compute_wer(["A", "B"], ["A"])


def _logits_for(emission: list[str], vocab: dict[str, int]) -> torch.Tensor:
    ids = [vocab[token] for token in emission]
    logits = torch.full((1, len(ids), len(vocab)), -10.0)
    for position, token_id in enumerate(ids):
        logits[0, position, token_id] = 10.0
    return logits


def test_greedy_decode_collapses_repeats_and_blanks(processor, vocab):
    """CTC emissions repeat characters and interleave blanks; decoding undoes both."""
    # H E E <pad> L L <pad> L O  ->  "HELLO"
    logits = _logits_for(["H", "E", "E", "<pad>", "L", "L", "<pad>", "L", "O"], vocab)

    assert greedy_decode(logits, processor) == ["HELLO"]
    assert greedy_decode(logits.numpy(), processor) == ["HELLO"]


def test_greedy_decode_strips_bos_eos(processor, vocab):
    """An undertrained model readily predicts <s>/</s>; they must not leak
    into decoded text (see the regression test below for why the naive fix,
    skip_special_tokens=True, is wrong)."""
    logits = _logits_for(["<s>", "H", "</s>", "I", "<s>"], vocab)

    assert greedy_decode(logits, processor) == ["HI"]


def test_greedy_decode_bos_eos_removal_preserves_blank_separated_repeats(processor, vocab):
    """Regression test: skip_special_tokens=True also strips <pad> (the CTC
    blank), which merges blank-separated repeats that must stay distinct --
    "HELLO" would silently become "HELO". bos/eos removal must not do that."""
    logits = _logits_for(
        ["<s>", "H", "E", "E", "<pad>", "L", "L", "<pad>", "L", "O", "</s>"], vocab
    )

    assert greedy_decode(logits, processor) == ["HELLO"]


def test_greedy_decode_accepts_numpy_and_torch_identically(processor, vocab):
    rng = np.random.default_rng(0)
    logits = rng.standard_normal((2, 12, len(vocab))).astype("float32")
    assert greedy_decode(logits, processor) == greedy_decode(torch.from_numpy(logits), processor)
