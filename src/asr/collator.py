"""Batch collation for CTC training.

One function, two different padding schemes -- which is exactly why this is the
easiest place in a CTC pipeline to introduce a silent bug:

* **audio** pads to the batch maximum with zeros, plus an ``attention_mask`` so
  the encoder ignores the padding;
* **labels** pad with ``-100``, the sentinel ``CTCLoss`` treats as "not a
  target". Pad labels with the tokenizer's pad id instead and the model happily
  trains against a stream of blank targets.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

#: Sentinel the loss ignores. Must not collide with a real token id.
LABEL_PAD_ID = -100


@dataclass
class DataCollatorCTCWithPadding:
    """Collate variable-length audio and transcripts into a padded batch.

    Args:
        processor: a ``Wav2Vec2Processor`` supplying the feature extractor and tokenizer.
        padding: forwarded to both padders -- ``True`` pads to the longest item
            in the batch, which is what you want; a fixed length wastes compute.
    """

    processor: Any
    padding: bool | str = True

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        batch = self.processor.feature_extractor.pad(
            [{"input_values": f["input_values"]} for f in features],
            padding=self.padding,
            return_tensors="pt",
        )

        labels_batch = self.processor.tokenizer.pad(
            [{"input_ids": f["labels"]} for f in features],
            padding=self.padding,
            return_tensors="pt",
        )
        # Replace pad positions with the ignore sentinel; keep real tokens intact.
        batch["labels"] = labels_batch["input_ids"].masked_fill(
            labels_batch["attention_mask"].ne(1), LABEL_PAD_ID
        )
        return dict(batch)
