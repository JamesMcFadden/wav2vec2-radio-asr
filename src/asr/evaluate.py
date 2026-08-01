"""Evaluate a pretrained Wav2Vec2 CTC checkpoint on LibriSpeech and report WER.

Sanity target: ``facebook/wav2vec2-base-960h`` scores ~3.4% WER on test-clean.
If this script reports something far off that, the harness is wrong before the
model is -- which is the entire point of running it before any fine-tuning.
"""

from __future__ import annotations

import argparse
import time
from collections.abc import Iterator

import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

from .data import TARGET_SAMPLE_RATE, decode_audio, load_librispeech, normalize_text
from .metrics import compute_wer, greedy_decode

DEFAULT_MODEL = "facebook/wav2vec2-base-960h"


def pick_device(requested: str = "auto") -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


@torch.inference_mode()
def transcribe_batch(
    waveforms: list, model: Wav2Vec2ForCTC, processor: Wav2Vec2Processor, device: torch.device
) -> list[str]:
    """Greedily transcribe a batch of waveforms.

    Whether to pass ``attention_mask`` is *not* a free choice -- it has to match
    how the checkpoint was trained, which the feature extractor records in
    ``return_attention_mask``. The base 960h models were trained without one;
    handing them a mask anyway degrades output badly (long utterances decode to
    the empty string), which reads like a broken pipeline rather than a config
    mismatch. The large/lv60 checkpoints are the opposite and need it.
    """
    feature_extractor = processor.feature_extractor
    wants_mask = bool(feature_extractor.return_attention_mask)

    inputs = feature_extractor(
        waveforms,
        sampling_rate=TARGET_SAMPLE_RATE,
        return_tensors="pt",
        padding=True,
        return_attention_mask=wants_mask,
    )

    forward_kwargs = {"input_values": inputs["input_values"].to(device)}
    if wants_mask:
        forward_kwargs["attention_mask"] = inputs["attention_mask"].to(device)

    logits = model(**forward_kwargs).logits
    return greedy_decode(logits.float().cpu(), processor)


def iter_batches(
    dataset, batch_size: int, *, sort_by_length: bool, buffer_batches: int = 16
) -> Iterator[tuple[list, list[str]]]:
    """Yield ``(waveforms, transcripts)`` batches.

    With ``sort_by_length`` the utterances are grouped by duration within a
    bounded buffer before batching. Padding is pure noise to a model that takes
    no attention mask, so putting similar-length clips together measurably cuts
    WER; a *global* sort would be marginally better still, but would mean
    decoding the whole split into memory first.

    Ordering is free to change because WER is computed corpus-wide.
    """
    chunk = max(1, batch_size * buffer_batches) if sort_by_length else batch_size

    for chunk_start in range(0, len(dataset), chunk):
        rows = dataset[chunk_start : chunk_start + chunk]
        items = [
            (decode_audio(entry), normalize_text(text))
            for entry, text in zip(rows["audio"], rows["text"], strict=True)
        ]
        if sort_by_length:
            items.sort(key=lambda item: len(item[0]))

        for start in range(0, len(items), batch_size):
            batch = items[start : start + batch_size]
            yield [audio for audio, _ in batch], [text for _, text in batch]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--config", default="clean", help="clean | other | all")
    parser.add_argument("--split", default="test")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--device", default="auto", help="auto | cpu | mps | cuda")
    parser.add_argument(
        "--dummy",
        action="store_true",
        help="use the 73-utterance smoke corpus instead of real LibriSpeech",
    )
    parser.add_argument("--show", type=int, default=3, help="print this many example transcripts")
    parser.add_argument(
        "--no-sort-by-length",
        dest="sort_by_length",
        action="store_false",
        help="disable length bucketing (slower and worse for mask-free models)",
    )
    args = parser.parse_args()

    device = pick_device(args.device)
    print(f"model  : {args.model}")
    print(f"device : {device}")

    processor = Wav2Vec2Processor.from_pretrained(args.model)
    model = Wav2Vec2ForCTC.from_pretrained(args.model).to(device).eval()

    dataset = load_librispeech(
        args.config, args.split, max_samples=args.max_samples, dummy=args.dummy
    )
    source = "librispeech_asr_dummy/validation" if args.dummy else f"{args.config}/{args.split}"
    print(f"data   : {source} ({len(dataset)} utterances)\n")

    predictions: list[str] = []
    references: list[str] = []
    started = time.perf_counter()

    for waveforms, transcripts in iter_batches(
        dataset, args.batch_size, sort_by_length=args.sort_by_length
    ):
        predictions.extend(transcribe_batch(waveforms, model, processor, device))
        references.extend(transcripts)
        print(f"\r  {len(references)}/{len(dataset)} utterances", end="", flush=True)

    elapsed = time.perf_counter() - started
    wer = compute_wer(predictions, references)

    print(f"\n\nWER    : {wer:.4f}  ({wer * 100:.2f}%)")
    print(f"time   : {elapsed:.1f}s")

    for reference, prediction in list(zip(references, predictions, strict=True))[: args.show]:
        print(f"\n  ref : {reference}")
        print(f"  hyp : {prediction}")


if __name__ == "__main__":
    main()
