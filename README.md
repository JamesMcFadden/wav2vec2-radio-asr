# wav2vec2-librispeech-asr

Fine-tuning and evaluating [Wav2Vec2.0](https://huggingface.co/docs/transformers/model_doc/wav2vec2)
for CTC speech recognition on [LibriSpeech](https://www.openslr.org/12).

Audio decodes through `soundfile` rather than the `datasets` `Audio` feature, so
the project has **no system dependencies** — no FFmpeg, no Homebrew.

## Setup

Requires [uv](https://docs.astral.sh/uv/). Python 3.12 and the venv are managed
for you:

```bash
git clone https://github.com/JamesMcFadden/wav2vec2-librispeech-asr
cd wav2vec2-librispeech-asr
uv sync
```

> **Keep this repo out of `~/Documents` and `~/Desktop`** if you use iCloud
> Desktop & Documents Sync. iCloud sets the `UF_HIDDEN` flag on files there, and
> Python ≥3.11 silently skips hidden `.pth` files — which breaks the editable
> install with a bare `ModuleNotFoundError: No module named 'asr'`.

## Evaluate a pretrained checkpoint

Start here. It validates the whole pipeline against a known number before any
training is attempted.

```bash
# 73-utterance smoke corpus, a few seconds
uv run python -m asr.evaluate --dummy

# real test-clean (~350 MB download on first run)
uv run python -m asr.evaluate --config clean --split test
```

### Results

| Model | Data | Batch | Device | WER |
|---|---|---|---|---|
| `facebook/wav2vec2-base-960h` | dummy (73 utts) | 1 | MPS | **5.30%** |
| `facebook/wav2vec2-base-960h` | dummy (73 utts) | 8, length-bucketed | MPS | **5.57%** |
| `facebook/wav2vec2-base-960h` | dummy (73 utts) | 8, unsorted | MPS | 6.43% |

Reproduce with:

```bash
uv run python -m asr.evaluate --dummy --batch-size 8
```

Full `test-clean` has not been run yet; the published reference for this
checkpoint is 3.4% WER, which is the number to check against.

## Two traps this repo already handles

**`attention_mask` must match the checkpoint.** The base 960h models were
trained without one. Passing a mask anyway takes WER from ~5% to **21%** and
decodes long utterances to the empty string — it looks like a broken pipeline,
not a config mismatch. `transcribe_batch` reads
`feature_extractor.return_attention_mask` and obeys it.

**Mixed-length batches cost accuracy.** Without an attention mask the encoder
treats zero-padding as audio, so batching utterances of very different lengths
degrades WER. Length bucketing (on by default) recovers most of it and is also
~40% faster. Disable with `--no-sort-by-length`.

## Layout

```
src/asr/
  data.py       LibriSpeech loading, soundfile decoding, transcript normalization
  collator.py   CTC collator -- audio pads with a mask, labels pad with -100
  metrics.py    greedy CTC decoding and jiwer WER
  evaluate.py   CLI entry point
tests/          offline by default; -m network for the Hub-dependent test
```

## Development

```bash
uv run pytest                   # full suite
uv run pytest -m 'not network'  # offline only
uv run ruff check . && uv run ruff format .
```

## Fine-tuning

Not implemented yet. When it lands it will start from `facebook/wav2vec2-base`,
build a character vocabulary, and freeze the feature encoder. Note that MPS
training is slow and CTC loss on MPS has historically been unstable — a rented
GPU or Colab is the realistic path for a full run.

## Licensing

Code is MIT (see [LICENSE](LICENSE)).

LibriSpeech is [CC BY 4.0](https://www.openslr.org/12). The
`facebook/wav2vec2-*` checkpoints are Apache 2.0.
