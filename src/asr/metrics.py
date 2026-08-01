"""Greedy CTC decoding and word error rate."""

from __future__ import annotations

from typing import Any

import jiwer
import numpy as np
import torch

from .collator import LABEL_PAD_ID


def greedy_decode(logits: torch.Tensor | np.ndarray, processor: Any) -> list[str]:
    """Argmax over the vocabulary, then collapse CTC repeats and blanks."""
    if isinstance(logits, torch.Tensor):
        predicted_ids = logits.argmax(dim=-1)
    else:
        predicted_ids = np.argmax(logits, axis=-1)
    return processor.batch_decode(predicted_ids)


def decode_labels(label_ids: torch.Tensor | np.ndarray, processor: Any) -> list[str]:
    """Decode reference labels, undoing the ``-100`` padding first.

    ``group_tokens=False`` is essential here: labels are a literal character
    sequence, not a CTC emission, so collapsing repeats would turn SEE into SE.
    """
    if isinstance(label_ids, torch.Tensor):
        label_ids = label_ids.detach().cpu().numpy()
    label_ids = np.asarray(label_ids).copy()
    label_ids[label_ids == LABEL_PAD_ID] = processor.tokenizer.pad_token_id
    return processor.tokenizer.batch_decode(label_ids, group_tokens=False)


def compute_wer(predictions: list[str], references: list[str]) -> float:
    """Corpus-level WER over aligned prediction/reference lists.

    Pairs with an empty reference are dropped -- jiwer cannot score them (the
    denominator is zero) and LibriSpeech occasionally yields one after
    normalization.
    """
    if len(predictions) != len(references):
        raise ValueError(f"length mismatch: {len(predictions)} predictions, {len(references)} refs")

    pairs = [(p, r) for p, r in zip(predictions, references, strict=True) if r.strip()]
    if not pairs:
        raise ValueError("no non-empty references to score against")

    hypotheses, refs = (list(x) for x in zip(*pairs, strict=True))
    return float(jiwer.wer(refs, hypotheses))


def build_compute_metrics(processor: Any):
    """Return a ``compute_metrics`` callable for ``transformers.Trainer``."""

    def compute_metrics(pred: Any) -> dict[str, float]:
        predictions = greedy_decode(pred.predictions, processor)
        references = decode_labels(pred.label_ids, processor)
        return {"wer": compute_wer(predictions, references)}

    return compute_metrics
