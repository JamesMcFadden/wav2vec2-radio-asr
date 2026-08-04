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
- **`aten::_ctc_loss` has no MPS kernel at all** — a hard `NotImplementedError`,
  not just instability. `asr/__init__.py` sets `PYTORCH_ENABLE_MPS_FALLBACK=1`
  as its first statement, before any of its own imports pull in torch, because
  the MPS fallback dispatch key reads that env var once at torch's static init,
  not per-call. Setting it later (e.g. inside `finetune.py`'s `main()`) is too
  late and the error comes back.
- **Load and `.map()` the dataset before touching CUDA, not after.**
  `datasets.map()` forks a worker subprocess even at `num_proc=1`. Forking a
  process that has already initialized a CUDA context (e.g. via
  `pick_device()`'s `torch.cuda.is_available()`, or loading a model onto the
  GPU) reliably deadlocks the child — it hangs at 0% progress indefinitely,
  burning near-zero CPU, with no exception raised. `finetune.py`'s `main()`
  does all data loading and `.map()` calls first and only calls
  `pick_device()`/`build_model()` afterward; don't reorder that. Confirmed by
  running the identical `.map()` call in isolation (no CUDA touched) vs. after
  `pick_device()` — only the latter hangs. `py-spy` cannot diagnose this
  directly in a RunPod pod: ptrace is blocked (`Permission denied`) even as
  root.
- **Don't use `load_best_model_at_end` for CTC fine-tuning at all.** Neither
  `metric_for_best_model="wer"` nor `"loss"` is safe: WER is coarse and
  readily ties across epochs (pinning whichever checkpoint hit the tied
  value first, since `Trainer` only updates "best" on a *strict*
  improvement), and CTC's `eval_loss` can silently diverge from `eval_wer`
  entirely -- loss can rise for many epochs while greedy-decoded WER keeps
  improving, because loss scores the full alignment distribution while WER
  only scores the single argmax path. Either metric has, in practice,
  caused `trainer.save_model("final")` to silently save a worse,
  non-final checkpoint with no error. `finetune.py` instead always keeps the
  literal last-step checkpoint and prints the best-by-`eval_wer` epoch from
  the log history as an informational reminder, not an automatic swap. See
  `FINETUNE_DEBUG_LOG.md`, 2026-08-03 and 2026-08-04.

## Environment

- macOS, Apple Silicon. Device is **MPS**, not CUDA. No `.cuda()`, no bf16.
- `uv` manages Python 3.12 and the venv. Use `uv run <cmd>`, not bare `python`.
- Training on MPS is slow, and the CTC loss op specifically falls back to CPU
  (see the `aten::_ctc_loss` constraint above) — fine for the `--dummy`
  smoke test, not for a real run. Use Colab or a rented GPU for that; CUDA has
  a native `ctc_loss` kernel.

## Data

- Never download the full 960h set unless explicitly asked. Default to
  `train.100`; use `--dummy` (73 utterances) for smoke tests.
- Config/split names are not uniform: `clean` and `other` use bare split names
  (`test`, `validation`, `train.100`), `all` uses dotted ones (`test.clean`).
  `load_librispeech` validates this before downloading.
- Cache stays in the `datasets` default cache. `data/` and `outputs/` are
  gitignored — never commit audio or checkpoints.

## Fine-tuning

Implemented in `asr/finetune.py`. `facebook/wav2vec2-base` (no CTC head) plus
the fixed vocab in `asr/vocab.py`, `model.freeze_feature_encoder()`, trained
with `transformers.Trainer`.

- Overfit 8 samples to ~100% train accuracy first:
  `uv run python -m asr.finetune --dummy --epochs 100 --batch-size 8`. CTC
  emits empty strings for the first few hundred steps — that is expected, not
  a bug. Still blank well past that means the LR or blank-token config is
  wrong.
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
