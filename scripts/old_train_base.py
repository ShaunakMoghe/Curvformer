import argparse, numpy as np
from datasets import load_dataset
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                          TrainingArguments, Trainer)

parser = argparse.ArgumentParser()
parser.add_argument("--model", default="bert-base-uncased")
parser.add_argument("--task", default="sst2")
parser.add_argument("--epochs", type=int, default=3)
parser.add_argument("--bsz", type=int, default=32)
parser.add_argument("--lr", type=float, default=2e-5)
args = parser.parse_args()

ds = load_dataset("glue", args.task)
tok = AutoTokenizer.from_pretrained(args.model)

def tok_fn(batch):
    text = batch["sentence"] if args.task=="sst2" else batch["text"]
    return tok(text, truncation=True, padding="max_length", max_length=128)
ds = ds.map(tok_fn, batched=True)

columns = ["input_ids", "attention_mask", "label"]
ds = ds.map(lambda e: {k: e[k] for k in columns}, batched=True)
ds.set_format(type="torch", columns=columns)

model = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=2)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {"accuracy": (preds==labels).mean()}

args_train = TrainingArguments(
    output_dir="./experiments/runs/baseline",
    per_device_train_batch_size=args.bsz,
    per_device_eval_batch_size=args.bsz,
    learning_rate=args.lr,
    num_train_epochs=args.epochs,
    bf16=True, tf32=True,
    dataloader_num_workers=2, dataloader_pin_memory=True,
    report_to=[],  # disable W&B
)

trainer = Trainer(
    model=model,
    args=args_train,
    train_dataset=ds["train"],
    eval_dataset=ds["validation"],
    compute_metrics=compute_metrics,
)

trainer.train()
trainer.evaluate()
