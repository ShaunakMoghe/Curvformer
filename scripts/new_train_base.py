import os, argparse
import torch
from datasets import load_dataset
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                          TrainingArguments, Trainer)
import numpy as np
import torch.nn.functional as F

_calls = {"n": 0}
_orig = F.scaled_dot_product_attention

def _probe_sdpa(q, k, v, *args, **kw):
    _calls["n"] += 1
    return _orig(q, k, v, *args, **kw)

F.scaled_dot_product_attention = _probe_sdpa


# --- fast matmul on 40-series (TF32) ---
torch.set_float32_matmul_precision("high")
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
try:
    torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=False, enable_mem_efficient=True)
except Exception:
    pass

parser = argparse.ArgumentParser()
parser.add_argument("--model", default="bert-base-uncased")
parser.add_argument("--task", default="sst2")
parser.add_argument("--epochs", type=int, default=3)
parser.add_argument("--bsz", type=int, default=32)
parser.add_argument("--lr", type=float, default=2e-5)
parser.add_argument("--max_length", type=int, default=128)
parser.add_argument("--train_layers", default="", help='Comma list of encoder layer indices to train (e.g. "10,11"). Empty = train all.')
parser.add_argument("--report_to", default="none", help='Logging integration: "none", "wandb", "tensorboard", etc.')
parser.add_argument("--seed", type=int, default=42)
args = parser.parse_args()

# --- data ---
ds = load_dataset("glue", args.task)
label_key = "label"

tok = AutoTokenizer.from_pretrained(args.model, use_fast=True)
field = "sentence" if args.task == "sst2" else "text"

def tok_fn(batch):
    return tok(batch[field], truncation=True, padding="max_length", max_length=args.max_length)

ds = ds.map(tok_fn, batched=True)
columns = ["input_ids", "attention_mask", label_key]
ds.set_format(type="torch", columns=columns)

# --- model ---
num_labels = int(ds["train"].features[label_key].num_classes or 2)
model = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=num_labels)

def set_trainable_layers(model, train_layers_str: str):
    """
    Freeze everything, then unfreeze classifier and the specified encoder blocks.
    Works for BERT- and RoBERTa-style encoder stacks.
    """
    if not train_layers_str:
        return  # train all params (default HF behavior)

    keep = {int(x) for x in train_layers_str.split(",")}
    # 1) freeze all
    for p in model.parameters():
        p.requires_grad = False

    # 2) unfreeze classifier head
    if hasattr(model, "classifier"):
        for p in model.classifier.parameters():  # BERT/Roberta classifiers
            p.requires_grad = True
    elif hasattr(model, "score"):  # some models name it 'score'
        for p in model.score.parameters():
            p.requires_grad = True

    # 3) unfreeze requested encoder blocks
    enc_layers = None
    if hasattr(model, "bert"):
        enc_layers = model.bert.encoder.layer
    elif hasattr(model, "roberta"):
        enc_layers = model.roberta.encoder.layer
    elif hasattr(model, "deberta"):
        enc_layers = model.deberta.encoder.layer
    else:
        raise RuntimeError("Unsupported model family for --train_layers.")

    for i in keep:
        for p in enc_layers[i].parameters():
            p.requires_grad = True

set_trainable_layers(model, args.train_layers)

# --- metrics ---
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    acc = (preds == labels).mean()
    return {"accuracy": float(acc)}

# --- training args ---
run_tag = f"layers_{args.train_layers or 'all'}"
args_train = TrainingArguments(
    output_dir=f"./experiments/runs/baseline_{run_tag}",
    per_device_train_batch_size=args.bsz,
    per_device_eval_batch_size=args.bsz,
    learning_rate=args.lr,
    num_train_epochs=args.epochs,
    fp16=True,
    logging_steps=50,
    seed=args.seed,
    # avoid newer fields like evaluation_strategy/report_to/save_strategy
)

trainer = Trainer(
    model=model,
    args=args_train,
    train_dataset=ds["train"],
    eval_dataset=ds["validation"],
    compute_metrics=compute_metrics,
)

trainer.train()
metrics = trainer.evaluate()
print("Final eval:", metrics)
print("SDPA calls in baseline:", _calls["n"])
steps = trainer.state.global_step
tok_per_step = args.bsz * 128
print("approx tokens/sec:", (steps * tok_per_step) / trainer.state.train_runtime)