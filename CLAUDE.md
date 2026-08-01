# CLAUDE.md

Wav2Vec2.0 CTC speech recognition, fine-tuned on ATC radio-channel audio and
validated against LibriSpeech.

## Purpose and framing

The end goal of this project is fine-tuning wav2vec2 on radio-channel ATC
audio (UWB-ATCC + ATCOSIM), not just evaluating a pretrained checkpoint on
LibriSpeech — LibriSpeech is the pipeline-validation baseline, ATC is the
point. This recreates the problem shape of a defense-sector ASR project from
a prior internship, using public substitute data since the original data and
code cannot be shared. Never describe specifics of that original project
(employer, scope, findings) in code, comments, commits, or docs — only the
generic framing above.

## Task

This is **CTC ASR**, not classification. Use `Wav2Vec2ForCTC` with
`Wav2Vec2Processor`. Never `Wav2Vec2ForSequenceClassification`.

## Verify before claiming

- Run `uv run pytest` before saying anything works. The fast suite is offline;
  `-m network` covers the Hub-dependent tests.
- Run `uv run ruff check . && uv run ruff format .` before committing.
- Sanity target: `facebook/wav2vec2-base-960h` scores ~3.4% WER on test-clean.
  A number far off that means the harness is broken, not the model.

## Non-obvious constraints

These are all things that have already cost time here:

- **`attention_mask` is not optional-by-taste.** Pass it only when the feature
  extractor says `return_attention_mask == True`. The base 960h checkpoints were
  trained without one; passing it anyway sends WER from ~5% to ~21% and decodes
  long utterances to the empty string. `transcribe_batch` reads the flag —
  don't hardcode it.
- **Length-bucket before batching.** Mask-free models see zero-padding as real
  audio, so mixed-length batches lose accuracy (6.43% → 5.57% on the smoke
  corpus just from sorting). Bucketing is also faster.
- **Labels pad with `-100`; audio pads with 0 + attention mask.** Two different
  schemes in `DataCollatorCTCWithPadding`. Don't unify them.
- **The tokenizer's pad token *is* the CTC blank** (id 0). Do not change it.
- **Decode reference labels with `group_tokens=False`.** Labels are a literal
  character sequence, not a CTC emission; collapsing repeats turns SEE into SE.
- **Audio decodes through `soundfile`, not `datasets`' `Audio` feature.** The
  latter now needs `torchcodec` + a system FFmpeg, which this project
  deliberately does not depend on. Keep `decode=False` on the column.
- **Keep this repo out of iCloud-synced folders** (`~/Documents`, `~/Desktop`).
  iCloud sets `UF_HIDDEN` on files there, and Python ≥3.11 silently skips hidden
  `.pth` files, which breaks the editable install with a bare `ModuleNotFoundError`.

## Environment

- macOS, Apple Silicon. Device is **MPS**, not CUDA. No `.cuda()`, no bf16.
- `uv` manages Python 3.12 and the venv. Use `uv run <cmd>`, not bare `python`.
- Training on MPS is slow and CTC loss there has historically been unstable.
  Prefer Colab or a rented GPU for real fine-tuning runs.

## Data

- Never download the full 960h set unless explicitly asked. Default to
  `train.100`; use `--dummy` (73 utterances) for smoke tests.
- Config/split names are not uniform: `clean` and `other` use bare split names
  (`test`, `validation`, `train.100`), `all` uses dotted ones (`test.clean`).
  `load_librispeech` validates this before downloading.
- Cache stays in the `datasets` default cache. `data/` and `outputs/` are
  gitignored — never commit audio or checkpoints.

## Fine-tuning, when we get there

- Start from `facebook/wav2vec2-base` (no CTC head) and build a char vocab.
- Call `model.freeze_feature_encoder()`.
- Overfit 8 samples to ~100% train accuracy first. CTC emits empty strings for
  the first few hundred steps — that is expected, not a bug. Still blank after
  ~500 steps on 8 samples means the LR or blank-token config is wrong.
- Training data: UWB-ATCC (~10.4h train) + ATCOSIM (~8h train), both free.
  Hold out the free ATCO2-test-set-1h as an extra out-of-domain eval — it's
  not from either training corpus.
- Benchmark target, not a reproduction requirement: Zuluaga-Gomez et al.,
  "How Does Pre-trained Wav2Vec 2.0 Perform on Domain Shifted ASR?" (IEEE SLT
  2022, arXiv:2203.16822) fine-tuned `wav2vec2-large-960h-lv60-self` on this
  same UWB-ATCC + ATCOSIM combination and reported ~10.5% WER on the joint
  eval set, down from ~18-19% pre-fine-tune on the individual sets. Their run
  used different hardware/hyperparameters and possibly LM rescoring for some
  numbers — treat this as a sanity-check ballpark, not a pass/fail bar.
