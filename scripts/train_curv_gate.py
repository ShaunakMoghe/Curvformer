# --- imports ---
import argparse
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from curvformer.patching import patch_bert_self_attention

# --- CurvTrainer subclass (can be anywhere above instantiation) ---
class CurvTrainer(Trainer):
    # Accept any extra kwargs HF may pass (e.g., num_items_in_batch)
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        outputs = model(**inputs)

        # Robustly extract loss from HF outputs
        if isinstance(outputs, dict) and "loss" in outputs:
            loss = outputs["loss"]
        elif hasattr(outputs, "loss"):
            loss = outputs.loss
        elif isinstance(outputs, tuple):
            loss = outputs[0]
        else:
            raise RuntimeError("Could not find loss in model outputs.")

        # Add curvature regularizer (if enabled)
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
ap.add_argument("--layers", default="all")  # e.g., "8,9,10,11" or "all"
args = ap.parse_args()

# --- data ---
ds = load_dataset("glue", args.task)
label_key = "label"
tok = AutoTokenizer.from_pretrained("bert-base-uncased")
def tok_fn(batch):
    key = "sentence" if args.task == "sst2" else "text"
    return tok(batch[key], truncation=True, padding="max_length", max_length=128)
ds = ds.map(tok_fn, batched=True)
cols = ["input_ids", "attention_mask", label_key]
ds = ds.map(lambda e: {k: e[k] for k in cols}, batched=True)
ds.set_format(type="torch", columns=cols)

# --- model (DEFINE IT BEFORE TRAINER) ---
model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
model = patch_bert_self_attention(model, layers=args.layers)  # uses your CurvSelfAttention

# --- metrics ---
import numpy as np
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {"accuracy": (preds == labels).mean()}

# --- training args ---
training_args = TrainingArguments(
    output_dir="./experiments/runs/curv",
    per_device_train_batch_size=args.bsz,
    per_device_eval_batch_size=args.bsz,
    learning_rate=args.lr,
    num_train_epochs=args.epochs,
    eval_strategy="epoch",
    save_strategy="no",
    fp16=True,
    logging_steps=50,
)

# --- NOW instantiate trainer ---
trainer = CurvTrainer(
    model=model,
    args=training_args,
    train_dataset=ds["train"],
    eval_dataset=ds["validation"],
    compute_metrics=compute_metrics,
)

# --- train/eval ---
trainer.train()
trainer.evaluate()
