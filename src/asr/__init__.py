"""Wav2Vec2.0 CTC speech recognition, fine-tuned on ATC radio-channel audio."""

import os

# Must be set before torch's first import, not merely before the op that needs
# it: the MPS fallback dispatch key is registered once at torch's static init
# and reads this env var then, not per-call. Set here because this is the
# first thing anything in the package imports, ahead of collator.py/metrics.py
# pulling in torch. Harmless off-MPS -- it only affects ops missing an MPS
# kernel, of which aten::_ctc_loss (used by fine-tuning) is one.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

from .atc_data import load_atc_corpus, load_finetune_eval, load_finetune_train  # noqa: E402
from .collator import LABEL_PAD_ID, DataCollatorCTCWithPadding  # noqa: E402
from .data import (  # noqa: E402
    TARGET_SAMPLE_RATE,
    decode_audio,
    load_librispeech,
    normalize_text,
    prepare_example,
)
from .metrics import build_compute_metrics, compute_wer, decode_labels, greedy_decode  # noqa: E402
from .vocab import CTC_VOCAB, write_vocab_json  # noqa: E402

__all__ = [
    "CTC_VOCAB",
    "LABEL_PAD_ID",
    "TARGET_SAMPLE_RATE",
    "DataCollatorCTCWithPadding",
    "build_compute_metrics",
    "compute_wer",
    "decode_audio",
    "decode_labels",
    "greedy_decode",
    "load_atc_corpus",
    "load_finetune_eval",
    "load_finetune_train",
    "load_librispeech",
    "normalize_text",
    "prepare_example",
    "write_vocab_json",
]
