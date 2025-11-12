# scripts/train_curv_gate.py  (DROP-IN)

# --- make ./src importable BEFORE any package imports ---
import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

# --- imports ---
import argparse
import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from curvformer.patching import patch_bert_self_attention


import torch.nn.functional as F

_calls = {"n": 0}
_orig = F.scaled_dot_product_attention

def _probe_sdpa(q, k, v, *args, **kw):
    _calls["n"] += 1
    return _orig(q, k, v, *args, **kw)

F.scaled_dot_product_attention = _probe_sdpa


# --- CurvTrainer: add optional regularizer safely ---
class CurvTrainer(Trainer):
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

# --- args ---
ap = argparse.ArgumentParser()
ap.add_argument("--task", default="sst2")
ap.add_argument("--epochs", type=int, default=1)
ap.add_argument("--bsz", type=int, default=64)
ap.add_argument("--lr", type=float, default=2e-5)
ap.add_argument("--layers", default="all")  # e.g. "8,9,10,11" or "all"
args = ap.parse_args()

# --- data ---
ds = load_dataset("glue", args.task)
label_key = "label"
tok = AutoTokenizer.from_pretrained("bert-base-uncased", use_fast=True)
field = "sentence" if args.task == "sst2" else "text"
def tok_fn(batch):
    return tok(batch[field], truncation=True, padding="max_length", max_length=128)
ds = ds.map(tok_fn, batched=True)
columns = ["input_ids", "attention_mask", label_key]
ds.set_format(type="torch", columns=columns)

# --- model & patch ---
model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
model = patch_bert_self_attention(model, layers=args.layers)  # swaps in CurvSelfAttention

# --- metrics ---
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {"accuracy": float((preds == labels).mean())}

# --- training args (compatible with older Transformers) ---
train_args = TrainingArguments(
    output_dir="./experiments/runs/curv",
    per_device_train_batch_size=args.bsz,
    per_device_eval_batch_size=args.bsz,
    learning_rate=args.lr,
    num_train_epochs=args.epochs,
    fp16=True,
    logging_steps=50,
)

# --- train/eval ---
trainer = CurvTrainer(
    model=model,
    args=train_args,
    train_dataset=ds["train"],
    eval_dataset=ds["validation"],
    compute_metrics=compute_metrics,
)
trainer.train()
print(trainer.evaluate())
print("SDPA calls in baseline:", _calls["n"])
steps = trainer.state.global_step
tok_per_step = args.bsz * 128
print("approx tokens/sec:", (steps * tok_per_step) / trainer.state.train_runtime)