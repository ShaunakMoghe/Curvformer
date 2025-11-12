import os, sys, argparse, random
import numpy as np
import torch

# Ensure we can import from ./src
THIS_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(THIS_DIR, ".."))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)

from curvformer.patching import patch_bert_self_attention  # expects src/curvformer/*

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def parse_layers_arg(layers_arg: str, num_layers: int):
    """
    Accepts:
      - "all"
      - comma list like "8,9,10,11,12" (1-based or 0-based).
    Returns a sorted list of 0-based layer indices within [0, num_layers-1].
    """
    if layers_arg.lower() == "all":
        return list(range(num_layers))
    raw = [s.strip() for s in layers_arg.split(",") if s.strip()]
    idxs = []
    for s in raw:
        i = int(s)
        # Heuristic: if given 1..12 for BERT-base, shift to 0..11
        if i >= num_layers:
            i = i - 1
        idxs.append(i)
    # clamp and dedupe
    idxs = sorted(set([max(0, min(num_layers - 1, i)) for i in idxs]))
    return idxs

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="bert-base-uncased")
    parser.add_argument("--task", default="sst2")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--bsz", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--max_len", type=int, default=128)
    parser.add_argument("--layers", default="all", help='e.g., "all" or "8,9,10,11,12"')
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)

    ds = load_dataset("glue", args.task)
    label_key = "label"

    tok = AutoTokenizer.from_pretrained(args.model)

    def tok_fn(batch):
        text_key = "sentence" if args.task == "sst2" else "text"
        return tok(
            batch[text_key],
            truncation=True,
            padding="max_length",
            max_length=args.max_len,
        )

    ds = ds.map(tok_fn, batched=True, remove_columns=[c for c in ds["train"].column_names if c not in ("label",)])
    columns = ["input_ids", "attention_mask", label_key]
    ds = ds.map(lambda e: {k: e[k] for k in columns}, batched=True)
    ds.set_format(type="torch", columns=columns)

    model = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=2)

    # Determine encoder depth and parse target layers robustly
    num_layers = model.config.num_hidden_layers
    target_layers = parse_layers_arg(args.layers, num_layers)

    # Patch in-place (uses CurvSelfAttention in src/curvformer/curv_attention.py)
    model = patch_bert_self_attention(model, layers=target_layers)

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        acc = (preds == labels).mean()
        return {"accuracy": float(acc)}

    training_args = TrainingArguments(
        output_dir="./experiments/runs/curv",
        per_device_train_batch_size=args.bsz,
        per_device_eval_batch_size=args.bsz,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        evaluation_strategy="epoch",
        save_strategy="no",
        logging_steps=50,
        fp16=torch.cuda.is_available(),
        report_to=["wandb"] if os.environ.get("WANDB_API_KEY") else [],
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ds["train"],
        eval_dataset=ds["validation"],
        compute_metrics=compute_metrics,
    )

    trainer.train()
    trainer.evaluate()

if __name__ == "__main__":
    main()
