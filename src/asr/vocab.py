"""The CTC character vocabulary shared by every checkpoint this repo trains.

Fixed rather than derived from a corpus: every dataset here (LibriSpeech, and
the ATC corpora once normalized by :func:`asr.data.normalize_text`) is already
restricted to A-Z, apostrophe, and spaces, so there's no benefit to scanning
text to discover a vocabulary that's known in advance -- and a fixed vocab
means checkpoints fine-tuned on different corpora stay compatible.
"""

from __future__ import annotations

import json
import string
from pathlib import Path

#: Blank/pad first (CTCLoss assumes blank id 0), then bos/eos, unk, the word
#: delimiter, and A-Z plus apostrophe. Order matters -- it fixes the token ids.
CTC_VOCAB_TOKENS: list[str] = ["<pad>", "<s>", "</s>", "<unk>", "|", *string.ascii_uppercase, "'"]

CTC_VOCAB: dict[str, int] = {token: index for index, token in enumerate(CTC_VOCAB_TOKENS)}


def write_vocab_json(path: Path) -> Path:
    """Write :data:`CTC_VOCAB` to ``path`` for ``Wav2Vec2CTCTokenizer.from_pretrained``."""
    path.write_text(json.dumps(CTC_VOCAB))
    return path
