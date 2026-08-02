"""Loading for the ATC fine-tuning corpora: ATCOSIM, UWB-ATCC, ATCO2-test-set-1h.

All three are published on the Hub by the SLT 2022 domain-shift paper's authors
with an identical schema (``id``, ``audio``, ``text``, ...), so one loader
covers all of them. Transcripts already arrive lowercase with numbers and
callsigns spelled out as words (e.g. "lufthansa four three nine three descend
to flight level two seven zero") -- no digits, no punctuation beyond spaces --
so :func:`asr.data.normalize_text` folds them into the CTC label space losslessly,
the same way it does LibriSpeech's.
"""

from __future__ import annotations

from datasets import Audio, Dataset, concatenate_datasets, load_dataset

#: Hub id and published splits for each corpus.
ATC_CORPORA: dict[str, dict[str, object]] = {
    "atcosim": {"hub_id": "Jzuluaga/atcosim_corpus", "splits": ("train", "test")},
    "uwb_atcc": {"hub_id": "Jzuluaga/uwb_atcc", "splits": ("train", "test")},
    "atco2_1h": {"hub_id": "Jzuluaga/atco2_corpus_1h", "splits": ("test",)},
}

#: The two free, freely-licensed corpora used for fine-tuning.
FINETUNE_CORPORA = ("atcosim", "uwb_atcc")

#: Held out entirely from training -- neither corpus above contributes to it --
#: so it doubles as an out-of-domain generalization check.
EVAL_CORPUS = "atco2_1h"


def load_atc_corpus(corpus: str, split: str, *, max_samples: int | None = None) -> Dataset:
    """Load one split of one ATC corpus, audio left undecoded (see :mod:`asr.data`)."""
    if corpus not in ATC_CORPORA:
        raise ValueError(f"unknown corpus {corpus!r}; expected one of {tuple(ATC_CORPORA)}")

    info = ATC_CORPORA[corpus]
    if split not in info["splits"]:
        raise ValueError(
            f"split {split!r} is not published for corpus {corpus!r}; "
            f"expected one of {info['splits']}"
        )

    dataset = load_dataset(info["hub_id"], split=split)
    dataset = dataset.cast_column("audio", Audio(decode=False))
    if max_samples is not None:
        dataset = dataset.select(range(min(max_samples, len(dataset))))
    return dataset


def load_finetune_train(
    *, corpora: tuple[str, ...] = FINETUNE_CORPORA, max_samples_per_corpus: int | None = None
) -> Dataset:
    """Concatenate the training splits of ``corpora`` into one shuffled dataset."""
    parts = [
        load_atc_corpus(corpus, "train", max_samples=max_samples_per_corpus) for corpus in corpora
    ]
    combined = concatenate_datasets(parts) if len(parts) > 1 else parts[0]
    return combined.shuffle(seed=0)


def load_finetune_eval(*, max_samples: int | None = None) -> Dataset:
    """Load the held-out out-of-domain eval split (ATCO2-test-set-1h)."""
    return load_atc_corpus(EVAL_CORPUS, "test", max_samples=max_samples)
