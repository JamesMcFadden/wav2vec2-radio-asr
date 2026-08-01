# wav2vec2-radio-asr

Fine-tuning [Wav2Vec2.0](https://huggingface.co/docs/transformers/model_doc/wav2vec2)
for CTC speech recognition on **radio-channel air traffic control (ATC) audio**,
with the evaluation pipeline validated against [LibriSpeech](https://www.openslr.org/12)
first.

Audio decodes through `soundfile` rather than the `datasets` `Audio` feature, so
the project has **no system dependencies** — no FFmpeg, no Homebrew.

## Motivation

This project recreates the problem shape of a defense-sector ASR project from a
2021 internship — fine-tuning speech recognition on noisy, jargon-heavy radio
chatter — using public substitute data, since the original data and code can't
be shared. ATC communications stand in for that domain: terse phraseology,
callsigns, and audio that has actually passed through a radio channel, rather
than clean studio speech like LibriSpeech.

## Plan

1. **Baseline (done):** validate the CTC evaluation pipeline against a known
   reference number on LibriSpeech, using `facebook/wav2vec2-base-960h`.
2. **Fine-tuning (in progress):** fine-tune wav2vec2 on ATC audio to
   demonstrate the pipeline handles a genuine domain shift, not just a clean
   read-speech corpus.

## Setup

Requires [uv](https://docs.astral.sh/uv/). Python 3.12 and the venv are managed
for you:

```bash
git clone https://github.com/JamesMcFadden/wav2vec2-radio-asr
cd wav2vec2-radio-asr
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

**Data:** [UWB-ATCC](https://lindat.mff.cuni.cz/repository/xmlui/handle/11858/00-097C-0000-0001-CCA1-0) (~10.4h train) +
[ATCOSIM](https://www.spsc.tugraz.at/databases-and-tools/atcosim-air-traffic-control-simulation-speech-corpus.html)
(~8h train) — both freely available. The free
[ATCO2-test-set-1h](https://www.atco2.org/data) is held out as an extra
out-of-domain eval set, since it isn't part of either training corpus.

**Benchmark target:** Zuluaga-Gomez et al., ["How Does Pre-trained Wav2Vec 2.0
Perform on Domain Shifted ASR? An Extensive Benchmark on Air Traffic Control
Communications"](https://arxiv.org/abs/2203.16822) (IEEE SLT 2022) fine-tuned
`wav2vec2-large-960h-lv60-self` on this same UWB-ATCC + ATCOSIM combination and
reported **~10.5% WER** on the joint eval set, down from ~18-19% pre-fine-tune
on the individual sets. This project targets comparable WER improvement as a
sanity check — their run used different hardware, hyperparameters, and
possibly LM rescoring for some numbers, so this isn't a claim of exact
reproduction.

## Licensing

Code is MIT (see [LICENSE](LICENSE)).

LibriSpeech is [CC BY 4.0](https://www.openslr.org/12). The
`facebook/wav2vec2-*` checkpoints are Apache 2.0. UWB-ATCC and ATCOSIM are
released for research use — see their respective pages above for exact terms.
