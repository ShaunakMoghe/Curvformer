# scripts/train_curv_gate.py — DROP-IN (single --layers controls patch+train)

# --- make ./src importable BEFORE any package imports ---
import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

# --- std imports ---
import argparse
import random
import numpy as np
import torch

from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)

from curvformer.patching import patch_bert_self_attention

# -------- SDPA probe (optional but handy) --------
import torch.nn.functional as F
_calls = {"n": 0}
_orig_sdpa = F.scaled_dot_product_attention
def _probe_sdpa(q, k, v, *args, **kw):
    _calls["n"] += 1
    return _orig_sdpa(q, k, v, *args, **kw)
F.scaled_dot_product_attention = _probe_sdpa
# -------------------------------------------------

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def parse_layers_arg(layers_arg: str, num_layers: int):
    """
    Accepts "all" or a comma list like "10,11".
    Returns sorted 0-based indices clamped to [0, num_layers-1].
    """
    if layers_arg.lower() == "all":
        return list(range(num_layers))
    raw = [s.strip() for s in layers_arg.split(",") if s.strip()]
    idxs = []
    for s in raw:
        i = int(s)
        # allow 1-based input by users: shift when needed
        if i >= num_layers:
            i = i - 1
        idxs.append(i)
    idxs = sorted(set(max(0, min(num_layers - 1, i)) for i in idxs))
    return idxs

def freeze_all_but(model, enc_layer_ids):
    """Freeze everything except classifier and the specified encoder blocks."""
    for p in model.parameters():
        p.requires_grad = False

    # unfreeze classifier
    for n, p in model.named_parameters():
        if n.startswith("classifier."):
            p.requires_grad = True

    # unfreeze chosen encoder layers
    for i in enc_layer_ids:
        pref = f"bert.encoder.layer.{i}."
        for n, p in model.named_parameters():
            if n.startswith(pref):
                p.requires_grad = True

def print_trainable(model):
    n_all = sum(p.numel() for p in model.parameters())
    n_trn = sum(p.numel() for p in model.parameters() if p.requires_grad)
    names = [n for n, p in model.named_parameters() if p.requires_grad][:12]
    print(f"[curv] Trainable params: {n_trn}/{n_all} ({100.0*n_trn/n_all:.2f}%)")
    if names:
        print("[curv] Sample trainable tensors:", names)

class CurvTrainer(Trainer):
    # Optional curvature regularizer hook (safe no-op if not used)
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        outputs = model(**inputs)
        loss = (
            outputs["loss"] if isinstance(outputs, dict) and "loss" in outputs
            else getattr(outputs, "loss", outputs[0])
        )
        reg_total = None
        for m in model.modules():
            if getattr(m, "reg_lambda", 0.0) > 0.0 and hasattr(m, "_curv_reg"):
                term = m.reg_lambda * m._curv_reg
                reg_total = term if reg_total is None else (reg_total + term)
        if reg_total is not None:
            loss = loss + reg_total
        return (loss, outputs) if return_outputs else loss

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="bert-base-uncased")
    ap.add_argument("--task", default="sst2")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--bsz", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--max_len", type=int, default=128)
    # SINGLE KNOB: controls both patching and training (old behavior)
    ap.add_argument("--layers", default="11", help='e.g. "11" or "10,11" or "all"')
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    set_seed(args.seed)

    # Data
    ds = load_dataset("glue", args.task)
    label_key = "label"
    tok = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    text_key = "sentence" if args.task == "sst2" else "text"

    def tok_fn(batch):
        return tok(
            batch[text_key],
            truncation=True,
            padding="max_length",
            max_length=args.max_len,
        )

    ds = ds.map(tok_fn, batched=True)
    cols = ["input_ids", "attention_mask", label_key]
    ds = ds.map(lambda e: {k: e[k] for k in cols}, batched=True)
    ds.set_format(type="torch", columns=cols)

    # Model + patch
    model = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=2)
    num_layers = model.config.num_hidden_layers
    target_layers = parse_layers_arg(args.layers, num_layers)

    # Patch attention in the selected blocks
    model = patch_bert_self_attention(model, layers=target_layers)

    # Freeze everything except classifier + those same blocks (old behavior)
    if args.layers.lower() != "all":
        freeze_all_but(model, target_layers)
    print_trainable(model)

    # Metrics
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        return {"accuracy": float((preds == labels).mean())}

    # Prefer bf16 on 40-series; otherwise fp16 if CUDA
    use_bf16 = torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8
    use_fp16 = torch.cuda.is_available() and not use_bf16

    train_args = TrainingArguments(
        output_dir="./experiments/runs/curv",
        per_device_train_batch_size=args.bsz,
        per_device_eval_batch_size=args.bsz,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        logging_steps=50,
        bf16=use_bf16,
        fp16=use_fp16,
        dataloader_num_workers=2,
        pin_memory=True,
        report_to=[],  # keep timing clean
    )

    trainer = CurvTrainer(
        model=model,
        args=train_args,
        train_dataset=ds["train"],
        eval_dataset=ds["validation"],
        compute_metrics=compute_metrics,
    )

    trainer.train()
    print(trainer.evaluate())

    print("SDPA calls in curv:", _calls["n"])
    steps = trainer.state.global_step or 1
    approx_tokens = steps * args.bsz * args.max_len
    rt = getattr(trainer.state, "train_runtime", None)
    if rt:
        print(f"approx tokens/sec: {approx_tokens/rt:.1f}")

if __name__ == "__main__":
    main()
