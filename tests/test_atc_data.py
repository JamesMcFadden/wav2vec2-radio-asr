"""Loading for the ATC fine-tuning and eval corpora.

Everything that touches the Hub is marked ``network`` -- run the fast subset
with ``uv run pytest -m 'not network'``.
"""

from __future__ import annotations

import pytest

from asr.atc_data import (
    EVAL_CORPUS,
    FINETUNE_CORPORA,
    load_atc_corpus,
    load_finetune_eval,
    load_finetune_train,
)
from asr.data import normalize_text


def test_unknown_corpus_rejected_before_download():
    with pytest.raises(ValueError, match="unknown corpus"):
        load_atc_corpus("not_a_real_corpus", "train")


def test_unknown_split_rejected_before_download():
    """atco2_1h only publishes a test split -- fail fast, not after a download."""
    with pytest.raises(ValueError, match="is not published for corpus"):
        load_atc_corpus("atco2_1h", "train")


def test_eval_corpus_is_excluded_from_finetuning():
    """The held-out eval set must never leak into training."""
    assert EVAL_CORPUS not in FINETUNE_CORPORA


@pytest.mark.network
def test_atcosim_loads_with_expected_columns():
    dataset = load_atc_corpus("atcosim", "test", max_samples=2)
    assert len(dataset) == 2
    assert {"id", "audio", "text"} <= set(dataset.column_names)


@pytest.mark.network
def test_atc_transcripts_normalize_without_dropping_words():
    """Digits arrive spelled out ('four three nine three'), so normalizing
    should never change the word count the way silently dropping a stray
    digit or symbol token would."""
    dataset = load_atc_corpus("atcosim", "test", max_samples=20)
    for text in dataset["text"]:
        assert len(normalize_text(text).split()) == len(text.split())


@pytest.mark.network
def test_load_finetune_train_combines_both_corpora():
    dataset = load_finetune_train(max_samples_per_corpus=3)
    assert len(dataset) == 3 * len(FINETUNE_CORPORA)


@pytest.mark.network
def test_load_finetune_eval_uses_atco2():
    dataset = load_finetune_eval(max_samples=2)
    assert len(dataset) == 2
