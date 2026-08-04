# Fine-tuning debug log

Running log of the fine-tuning investigation: what broke, what was tried, what
was found. Confirmed, generally-applicable fixes also get promoted into
CLAUDE.md's "Non-obvious constraints" section; this file is the messier
in-progress trail, kept so a resumed investigation doesn't repeat dead ends.

## 2026-08-03: Full 30-epoch run completed but never learned

**Symptom:** `uv run python -m asr.finetune` (defaults: `facebook/wav2vec2-base`,
LR `3e-4`, `warmup_steps=500`, 30 epochs) ran to completion on a RunPod A40 in
~7 hours with no crash, but `eval_wer` was exactly `1.0` at **every single
epoch from 1 through 30** — not just early, the whole run. Training loss
dropped from ~20 to ~3.9 within the first ~20-30 steps, then stayed flat
(3.25-3.32, pure noise, no trend) for the remaining ~70,980 steps.

**Diagnosis:** classic CTC collapse-to-blank. The model found a degenerate
local minimum (predicting silence/blank for everything, which trivially lowers
CTC loss) within the first few dozen steps and never escaped it. The
`--dummy` smoke tests run earlier (2 and 5 epochs) also showed `eval_wer: 1`,
which was read as "expected, not enough steps yet" per the documented
CTC-emits-blank-early behavior — in hindsight that was a warning sign that
went unchecked. The CLAUDE.md-prescribed sanity check (overfit 8 samples to
~100% train accuracy) was never actually run to completion before committing
to the full run.

**Cost of not catching this earlier:** ~7 GPU-hours (~$3) and a day of wall
time. Lesson: always finish the overfit sanity check first, and check *actual
decoded predictions*, not just whether the eval WER metric moves.

## 2026-08-03: Overfit diagnostic with lower LR — better loss, still not learning

**Change tried:** `--dummy --epochs 100 --batch-size 8 --learning-rate 5e-5
--warmup-steps 10` (peak LR down 6x from default, warmup short enough to
actually complete within the 200-step dummy budget — the default
`warmup_steps=500` was longer than the entire dummy run, so LR never
meaningfully ramped in the earlier smoke tests either).

**Result:** train loss decreased *monotonically* the whole run (19 → 2.83),
a qualitatively different and better trajectory than the full run's instant
flatline. But that alone doesn't mean it overfit — `eval_wer` isn't the right
signal to watch here since eval is the held-out ATCO2 set, unrelated to the
16 training examples. Checked actual greedy-decoded predictions against the
training references directly (`check_overfit.py`, run against the saved
`/tmp/diag/final` checkpoint): **0/16 exact matches**, and the hypotheses
don't even loosely resemble the references — garbled, repetitive sequences
dominated by a handful of letters (P, W, Z, U, B, H), regardless of what the
reference text says.

**Two things found while looking at this:**
1. **Bug, confirmed:** `greedy_decode` (`src/asr/metrics.py`) calls
   `processor.batch_decode(predicted_ids)` without `skip_special_tokens=True`.
   HF's default is `skip_special_tokens=False`, so `<s>`/`</s>`/`<unk>` get
   decoded as literal text and pollute every prediction. Doesn't fully explain
   the failure (the actual letters predicted still don't track the
   references), but is actively corrupting what we can see and needs fixing
   regardless.
2. **Hypothesis, not yet confirmed:** CTC output-timestep count vs. label
   length. Some training references are long (~100+ characters, e.g. "AUSTRIAN
   THREE TWO THREE GOLF CLIMB FLIGHT LEVEL THREE FOUR ZERO LEVEL THREE FOUR
   ZERO AUSTRIAN THREE TWO THREE G"). wav2vec2's conv feature encoder
   downsamples audio by ~320x; if a clip's downsampled timestep count is close
   to or below its label length, CTC alignment becomes infeasible and
   `ctc_zero_infinity=True` (set deliberately to stop one bad batch from
   poisoning training) silently zeroes the loss/gradient for that example
   instead of raising an error. If this is happening for a meaningful fraction
   of examples, the model would only ever get real gradient signal from a
   subset, which could explain slow/absent convergence on word content even
   as overall loss drops. **Next step: check this directly before trying
   another longer overfit run.**

## 2026-08-03: Decode bug fixed, length-mismatch hypothesis refuted

**Decode fix:** `skip_special_tokens=True` (tried first) was itself wrong --
`<pad>` is also a "special token" in this tokenizer, and stripping it before
the CTC-aware repeat-collapse step corrupts genuine double letters ("HELLO"
decodes to "HELO"). Confirmed with a manual tokenizer test before touching
the real code. Correct fix: remap `bos_token_id`/`eos_token_id` to
`pad_token_id` *before* decoding, so the existing (correct) CTC blank-handling
logic absorbs them instead of a separate skip-special-tokens code path.
`<unk>` is left alone -- it's a meaningful "model was unsure" signal, not
noise. Fixed in `greedy_decode` (`src/asr/metrics.py`), with two regression
tests pinning down both the bos/eos-stripping behavior and the
repeat-collapse-must-still-work invariant (`tests/test_metrics.py`). Landed
in commit history separately from this log.

**CTC length-mismatch hypothesis: refuted.** Checked
`model._get_feat_extract_output_lengths()` against label length for all 16
overfit examples: every one has 3-4x more output timesteps than its label
needs (e.g. 211 timesteps for a 62-character label). `ctc_zero_infinity`
zeroing gradients from a length mismatch is not what's happening here.

**Re-checked the earlier diagnostic's predictions with the fixed decoder**
(same `/tmp/diag/final` checkpoint, no retraining): cleaner output (no more
`<s>`/`</s>` litter), but still **0/16 exact matches**, and the actual
character content is still unrelated to the references -- same handful of
letters (P, W, Z, U, B, H...) dominating regardless of what the reference
says. So the decode bug was real but not the (or not the whole) cause.

**Working theory now:** 200 total optimizer steps (100 epochs x 2 steps/epoch
on 16 examples at batch size 8) is just not enough. wav2vec2-base has ~90M
*unfrozen* transformer parameters (only the CNN feature encoder is frozen)
plus a randomly-initialized CTC head that has to learn blank/repeat structure
from scratch -- this class of model typically needs thousands of steps to
overfit even a handful of examples, not hundreds. The diagnostic's LR
schedule also worked against it: linear decay was tied to the artificially
short 200-step total, so LR was heading toward ~0 right as loss was still
dropping steadily (2.98 -> 2.83 and still trending down when the run ended).

## 2026-08-03: Longer overfit run (2000 steps) -- theory confirmed

**Change tried:** same `--dummy --learning-rate 5e-5` as before, but
`--epochs 1000 --warmup-steps 20` instead of `100`/`10` -- 2000 total steps
instead of 200, and critically a much slower LR decay (tied to the longer
step count), so LR stays near its 5e-5 peak for far longer instead of racing
to zero.

**Result so far:** train loss broke well past the ~2.8 plateau the 200-step
run got stuck at -- it kept falling steadily past 1.0 (by epoch ~220) and was
at ~0.19 by epoch ~550 (step ~1100), still trending down. More importantly,
checked actual decoded predictions at an early-ish checkpoint
(`checkpoint-298`, ~149 epochs in) with `check_checkpoint.py`: predictions now
contain **genuine recognizable word fragments** in roughly the right places
-- "FOUR", "ONE", "TRA" (matching "TRASADINGEN") -- a completely different
character than the garbled, content-independent P/W/Z/U/B/H soup from the
200-step run. Not exact matches yet at that checkpoint, but clearly real
learning, confirming the working theory: **200 steps was just not enough**
for wav2vec2-base's unfrozen transformer body + random-init CTC head to
learn alignment, even on 16 examples. No further code bug needed here --
this was a training-budget problem, not a pipeline bug.

**Debugging note:** hit a red herring while checking intermediate
checkpoints -- `Wav2Vec2ForCTC.from_pretrained(checkpoint_dir)` failed with a
confusing `HFValidationError: Repo id must be in the form...` even when the
checkpoint directory clearly had all files present (`model.safetensors`,
`config.json`, etc). Root cause was mundane: grabbing the *most recent*
checkpoint directory raced against `Trainer`'s own checkpoint write (weights
file briefly exists under a `.tmp*` name before being renamed), so the
"empty"/malformed path fell through transformers' local-file resolution into
a Hub-repo-id code path that produced this misleading error. Fixed by
checking the *second*-most-recent checkpoint (guaranteed fully written, since
`save_total_limit=2` only rotates an older one out after a newer one finishes
writing) and by passing `local_files_only=True` to rule out any Hub-fallback
behavior. Worth remembering next time a local checkpoint path fails to load
with a Hub-shaped error message -- check whether it's still being written
before assuming the checkpoint itself is broken.

## 2026-08-03: 2000-step run finished -- overfit confirmed, but `final/` was stale

**Result: 11/16 exact matches** at the true end-of-training checkpoint
(`checkpoint-2000`, step 2000/2000), with the 5 misses all near-misses (one
or two characters off: "IDENTIFIED" -> "IDENTIED", "PRAHA RADAR" ->
"PRAHARADAR", "K OSCAR ZULU" -> "OSCAR ZULU"), not the garbled, content-free
output from earlier diagnostics. This conclusively confirms the working
theory from the previous entry: **the fine-tuning pipeline is not broken.**
Given enough optimizer steps and a sane learning rate, wav2vec2-base
correctly learns CTC alignment and transcription content. The original full
30-epoch run's collapse-to-blank was a hyperparameter problem (`lr=3e-4` too
aggressive for a randomly-initialized CTC head), not a code/data bug.

**New bug found while getting this result: `output_dir/final` was NOT the
step-2000 model.** Checking `output_dir/final` directly (as the training
script itself writes it) reproduced the *old* garbled, 0/16-match output --
identical to the much-earlier `checkpoint-298` diagnostic. Root cause:
`load_best_model_at_end=True` + `metric_for_best_model="wer"`, combined with
`eval_wer` being scored against the *held-out ATCO2 set* (irrelevant to an
overfit diagnostic, and unrelated to the 16 training examples). `eval_wer`
dipped to `0.9915` once, at epoch ~149 (checkpoint-298) -- almost certainly
noise -- then sat at exactly `1.0` for every one of the remaining ~850
epochs. `Trainer`'s "best" tracking only overwrites on a *strict*
improvement, so a tie never dethrones the first checkpoint that reached the
best value. `trainer_state.json` confirmed it directly:
`best_model_checkpoint: checkpoint-298`, `best_metric: 0.9915...`. Since
`load_best_model_at_end=True`, `trainer.train()` silently reloads that
pinned checkpoint into `trainer.model` before `main()`'s
`trainer.save_model(output_dir / "final")` call runs -- so `final/` ends up
being whatever checkpoint happened to eke out the first (possibly noise-driven)
tie-break, not the most-trained model. This is a **general risk for the real
run too**, not just this diagnostic: any real training run where `eval_wer`
plateaus or ties across epochs (plausible, since WER is coarse/discrete
relative to continuous loss) would silently save a worse-than-final model to
`final/` with no error or warning.

**Fix applied:** changed `metric_for_best_model` from `"wer"` to `"loss"`
(`src/asr/finetune.py`). `eval_loss` is continuous-valued and, in practice,
never exactly ties between epochs, so "best" tracking behaves as intended
(genuinely favors the better-generalizing checkpoint) without the discrete-tie
pinning trap that `wer` has. `load_best_model_at_end=True` itself is kept --
it's the right behavior for the real run, where overfitting to the training
distribution over 30 epochs is a real concern and we do want the
best-on-held-out-data checkpoint, not necessarily literally the last step.
Promoted to CLAUDE.md's "Non-obvious constraints" since this would silently
corrupt any future run's `final/` output the same way.

**Lesson for future diagnostics:** when sanity-checking with `--dummy`,
don't trust `output_dir/final` -- check the highest-numbered
`checkpoint-N/model.safetensors` directly, or (better, now that the metric
is fixed) rely on `eval_loss` doing the right thing.

## Open questions / next steps

- [x] Fix `skip_special_tokens=True` in `greedy_decode`.
- [x] Measure actual CTC output-timestep count vs. label length -- ruled out.
- [x] Re-run the overfit diagnostic with many more steps -- confirmed real
      learning is happening; loss well below the earlier plateau and
      predictions show genuine word fragments.
- [x] Let the 2000-step run finish; check exact-match rate at the end --
      **11/16 exact matches** at the true final checkpoint. Pipeline
      confirmed working.
- [x] Found and fixed a second real bug along the way: `load_best_model_at_end`
      + `metric_for_best_model="wer"` pinned a stale, barely-trained
      checkpoint into `final/` due to WER ties. Fixed by switching the
      tracked metric to `eval_loss`.
- [ ] Decide real-run hyperparameters: the defaults used for the failed
      30-epoch/71,010-step run (`lr=3e-4`, `warmup_steps=500`) caused instant
      collapse; `lr=5e-5` is now confirmed to actually learn. Plan: relaunch
      the full run with `lr=5e-5` (and proportionally longer warmup, e.g.
      ~1000-2000 steps out of 71,010 total, matching this diagnostic's
      warmup:total ratio) once the user approves spending more GPU time.
- [ ] Re-run (or at least re-verify with the fixed metric) the `--dummy`
      overfit sanity check before committing to the next full run, per
      CLAUDE.md's prescribed workflow -- this time trusting `final/`.

## 2026-08-04: Corrected full run launched -- learning confirmed early

Relaunched the full 30-epoch/71,010-step run with `lr=5e-5`,
`warmup_steps=1000` (vs. the failed run's `lr=3e-4`, `warmup_steps=500`),
into a fresh `--output-dir outputs/finetune_v2` so `get_last_checkpoint`
wouldn't resume from the collapsed run's weights. `finetune.py` was synced to
the pod first with the `metric_for_best_model="loss"` fix from the previous
entry.

Unlike the original run (`eval_wer` pinned at exactly `1.0` every epoch,
loss flatlined ~3.25-3.32 from step ~30 onward), this run shows real,
monotonic improvement from the start:

| epoch | eval_wer | eval_loss |
|-------|----------|-----------|
| 1     | 0.6803   | 1.651     |
| 2     | 0.5703   | 1.398     |
| 3     | 0.5381   | 1.370     |

Training loss is also declining steadily rather than flatlining (~2.9 at
step ~900 down to ~2.86 by step ~1100, out of 71,010 total). This is strong
early confirmation that the `lr=3e-4` default was the actual root cause of
the original failure, consistent with everything found in the overfit
diagnostics above.

## 2026-08-04: Run completed -- `metric_for_best_model="loss"` also pins the wrong checkpoint

**Full eval_wer trajectory** (epoch: wer): 1: .680, 2: .570, 3: .538, 4: .520,
5: .491, 6: .496, 7: .483, 8: .477, 9: .463, 10: .457, 11: .464, 12: .449,
13: .474, 14: .452, 15: .444, 16: .457, 17: .439, 18: .430, 19: .428,
20: .432, 21: .430, 22: .424, 23: .427, 24: .427, 25: .430, 26: .421,
27: .423, 28: .422, 29: .422, **30: .4185 (lowest of the run)**. Real,
monotonic-ish improvement the whole way, plateauing but still slightly
improving by epoch 30 -- a completely different shape than the original
run's flat `1.0` forever. `eval_loss`, however, does *not* track this:
it bottoms out around epoch 10 (`1.326`) and *rises* for the rest of the run
(up to `~1.75-1.79` by epoch 26-30) even as `eval_wer` keeps improving.

**Consequence:** `metric_for_best_model="loss"` (the previous entry's fix)
picked `checkpoint-23670` (epoch 10, `eval_wer=0.4566`) as "best" and
`load_best_model_at_end=True` silently reloaded it before
`trainer.save_model("final")` ran -- so `final/` held a *worse* checkpoint
than the actual last epoch (`eval_wer=0.4185`), for a different underlying
reason than the earlier WER-tie bug but the same failure shape: **a wrong,
non-final model silently written to `final/` with no error.** Caught by
spot-checking decoded predictions from `checkpoint-71010` (the true last
step) directly and finding they looked meaningfully better than what
`final/` produced, then confirming via `trainer_state.json`'s
`best_model_checkpoint`/`best_metric` fields and the full log history.

Manually verified `checkpoint-71010`'s predictions on 16 ATCO2 eval examples:
mostly correct, coherent ATC phraseology with minor letter-level errors
("RESTICTIONS" for "RESTRICTIONS", dropped/altered short words) --
qualitatively a working ASR model, not garbage. `compute_wer` on that
16-example sample: `0.2067` (sample noise vs. the full-set `0.4185`
figure is expected at n=16).

**This is CTC-specific and not a fluke:** CTC's loss is a soft, full-distribution
likelihood over all possible alignments, while WER only scores the single
greedy/argmax path. A model can become more confident about *wrong*
alignment mass elsewhere (raising loss) while its greedy decode keeps
getting more accurate (lowering WER) -- so loss and WER are not guaranteed
to co-move for CTC the way they typically do for, e.g., classification.
Neither `"wer"` (ties) nor `"loss"` (divergence from the metric that
actually matters) is safe as `metric_for_best_model` here.

**Fix applied:** removed `load_best_model_at_end`/`metric_for_best_model`
from `TrainingArguments` entirely. `final/` now always holds the literal
last-step checkpoint (what `trainer.train()` naturally ends on), which
matches how `save_strategy="epoch"` + `save_total_limit=2` already keeps the
last checkpoint around regardless. Added a post-training report that scans
`trainer.state.log_history` for the best `eval_wer` epoch and prints it
alongside a reminder to reload that specific `checkpoint-N/` directory
instead of `final/` if it turns out to matter (e.g. if a run overfits badly
late and regresses) -- informational, not automatic, since automatic
selection is exactly what just broke twice. Manually fixed this run's
already-wrong `outputs/finetune_v2/final` on the pod by copying
`checkpoint-71010`'s weights over it (old contents preserved at
`final_wrong_epoch10/` for reference).

**Bottom line for this run: pipeline and hyperparameters both confirmed
working.** Final (epoch 30, true last checkpoint) `eval_wer = 0.4185` on the
held-out ATCO2 set. Zuluaga-Gomez et al. reported ~10.5% WER on their joint
eval set, but with `wav2vec2-large-960h-lv60-self` (a much larger model with
far more pretraining data) rather than `wav2vec2-base` used here, and
possibly LM rescoring for some of their reported numbers -- so this gap is
expected, not a sign of a remaining bug. A stronger base checkpoint
(`wav2vec2-large-960h-lv60-self` or similar) is the natural next lever if
closing that gap further is a goal.
