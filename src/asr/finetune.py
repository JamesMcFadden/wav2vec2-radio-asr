"""Fine-tune Wav2Vec2.0 for CTC on ATC radio-channel audio.

Trains on ATCOSIM + UWB-ATCC (~18.4h combined, both free) and evaluates on the
held-out ATCO2-test-set-1h, which contributes to neither training corpus and
so also checks out-of-domain generalization.

Benchmark target, not a reproduction requirement: Zuluaga-Gomez et al., "How
Does Pre-trained Wav2Vec 2.0 Perform on Domain Shifted ASR?" (IEEE SLT 2022,
arXiv:2203.16822) fine-tuned wav2vec2-large-960h-lv60-self on this same
combination and reported ~10.5% WER on the joint eval set. Their run used
different hardware, hyperparameters, and possibly LM rescoring for some
numbers -- treat this as a ballpark, not a pass/fail bar.

Before a real run, sanity-check the harness by overfitting 8 samples:

    uv run python -m asr.finetune --dummy --epochs 100 --batch-size 8

CTC emits empty strings for the first few hundred steps -- expected, not a
bug. Still blank well past that on 8 samples means the LR or blank-token
config is wrong, not the data or model.

Re-running against the same --output-dir resumes from the latest checkpoint
automatically -- relevant on a rented GPU that can be reclaimed mid-run
(e.g. a Vast.ai interruptible instance). Point --output-dir at a network
volume that outlives the pod if you want that to actually survive a
reclaim; a container's local disk does not.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from transformers import (
    Trainer,
    TrainingArguments,
    Wav2Vec2CTCTokenizer,
    Wav2Vec2FeatureExtractor,
    Wav2Vec2ForCTC,
    Wav2Vec2Processor,
)
from transformers.trainer_utils import get_last_checkpoint

# Any import of this module first runs asr/__init__.py, which sets
# PYTORCH_ENABLE_MPS_FALLBACK before its own imports pull in torch -- see the
# comment there. That has to happen before torch's MPS backend initializes;
# setting it any later is too late for aten::_ctc_loss, which has no MPS
# kernel at all.
from .atc_data import load_finetune_eval, load_finetune_train
from .collator import DataCollatorCTCWithPadding
from .data import TARGET_SAMPLE_RATE, prepare_example
from .evaluate import pick_device
from .metrics import build_compute_metrics
from .vocab import write_vocab_json

DEFAULT_MODEL = "facebook/wav2vec2-base"
DUMMY_SAMPLES_PER_CORPUS = 8
DUMMY_EVAL_SAMPLES = 8


def build_processor(output_dir: Path) -> Wav2Vec2Processor:
    """Build a processor from the fixed CTC vocab.

    ``return_attention_mask=False`` matches how ``facebook/wav2vec2-base`` (as
    opposed to the large/lv60 checkpoints) expects to be fine-tuned -- see the
    ``attention_mask`` note in CLAUDE.md.
    """
    vocab_path = write_vocab_json(output_dir / "vocab.json")
    tokenizer = Wav2Vec2CTCTokenizer(
        str(vocab_path),
        unk_token="<unk>",
        pad_token="<pad>",
        word_delimiter_token="|",
    )
    feature_extractor = Wav2Vec2FeatureExtractor(
        feature_size=1,
        sampling_rate=TARGET_SAMPLE_RATE,
        padding_value=0.0,
        do_normalize=True,
        return_attention_mask=False,
    )
    return Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)


def build_model(model_name: str, processor: Wav2Vec2Processor) -> Wav2Vec2ForCTC:
    model = Wav2Vec2ForCTC.from_pretrained(
        model_name,
        ctc_loss_reduction="mean",
        # Without this, CTCLoss returns inf whenever a batch's predicted
        # timesteps are shorter than its target length -- common early in
        # training on short utterances -- and inf loss poisons the whole batch.
        ctc_zero_infinity=True,
        pad_token_id=processor.tokenizer.pad_token_id,
        vocab_size=len(processor.tokenizer),
    )
    model.freeze_feature_encoder()
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--output-dir", default="outputs/finetune")
    parser.add_argument("--epochs", type=float, default=30.0)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--warmup-steps", type=int, default=500)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-eval-samples", type=int, default=None)
    parser.add_argument("--num-proc", type=int, default=1, help="workers for dataset.map")
    parser.add_argument("--device", default="auto", help="auto | cpu | mps | cuda")
    parser.add_argument(
        "--dummy",
        action="store_true",
        help=f"overfit-8-samples smoke test: {DUMMY_SAMPLES_PER_CORPUS} examples per "
        f"training corpus, {DUMMY_EVAL_SAMPLES} eval examples",
    )
    args = parser.parse_args()

    if args.dummy:
        args.max_train_samples = args.max_train_samples or DUMMY_SAMPLES_PER_CORPUS
        args.max_eval_samples = args.max_eval_samples or DUMMY_EVAL_SAMPLES

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"model  : {args.model}")
    processor = build_processor(output_dir)

    print("loading data...")
    train_dataset = load_finetune_train(max_samples_per_corpus=args.max_train_samples)
    eval_dataset = load_finetune_eval(max_samples=args.max_eval_samples)
    print(f"train  : {len(train_dataset)} utterances (ATCOSIM + UWB-ATCC)")
    print(f"eval   : {len(eval_dataset)} utterances (ATCO2-test-set-1h, held out)")

    # dataset.map() spawns a worker subprocess even at num_proc=1. Forking a
    # process after CUDA has been initialized in the parent reliably deadlocks
    # the child (a well-known PyTorch + multiprocessing + fork hazard) -- so
    # every map() call has to happen before pick_device()/build_model() below
    # ever touch torch.cuda, not after.
    print("preparing features (this decodes and extracts every utterance once)...")
    train_dataset = train_dataset.map(
        prepare_example,
        fn_kwargs={"processor": processor},
        remove_columns=train_dataset.column_names,
        num_proc=args.num_proc,
        desc="train",
    )
    eval_dataset = eval_dataset.map(
        prepare_example,
        fn_kwargs={"processor": processor},
        remove_columns=eval_dataset.column_names,
        num_proc=args.num_proc,
        desc="eval",
    )

    device = pick_device(args.device)
    if device.type == "mps":
        # aten::_ctc_loss has no MPS kernel -- torch raises NotImplementedError,
        # not just slow/unstable results. PYTORCH_ENABLE_MPS_FALLBACK (set at
        # import time, above) routes that one op to CPU, which is what makes
        # the local --dummy smoke test runnable at all. A real run belongs on
        # CUDA (rented GPU/Colab), where ctc_loss is native.
        print("note   : aten::_ctc_loss has no MPS kernel; that op falls back to CPU")
    print(f"device : {device}")

    model = build_model(args.model, processor)

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        weight_decay=0.005,
        logging_steps=10,
        report_to=[],
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
        use_cpu=(device.type == "cpu"),
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=DataCollatorCTCWithPadding(processor=processor),
        compute_metrics=build_compute_metrics(processor),
        processing_class=processor,
    )

    last_checkpoint = get_last_checkpoint(str(output_dir))
    if last_checkpoint:
        print(f"resuming from {last_checkpoint}")
    trainer.train(resume_from_checkpoint=last_checkpoint)

    print("\nfinal eval:")
    print(trainer.evaluate())

    trainer.save_model(str(output_dir / "final"))
    processor.save_pretrained(str(output_dir / "final"))
    print(f"\nsaved to {output_dir / 'final'}")


if __name__ == "__main__":
    main()
