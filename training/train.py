"""
train.py
--------
Fine-tunes DistilBERT on the labeled deferment document dataset.

Steps:
  1. Load dataset CSV
  2. Tokenize text
  3. Split into train/test sets
  4. Fine-tune DistilBERT with HuggingFace Trainer
  5. Save model + tokenizer to models/

Usage:
    python train.py

After training, the model is saved to: models/distilbert-fraud-classifier/
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
)

sys.path.append(str(Path(__file__).parent))
import config


def load_data() -> tuple[list[str], list[int]]:
    """Load the labeled CSV and return (texts, labels)."""
    print(f"[1/5] Loading dataset from: {config.DATA_CSV}")
    df = pd.read_csv(config.DATA_CSV)
    df = df.dropna(subset=["text", "label"])
    df["label"] = df["label"].astype(int)
    print(f"      Total samples: {len(df)} | Genuine: {(df['label']==0).sum()} | Fraudulent: {(df['label']==1).sum()}")
    return df["text"].tolist(), df["label"].tolist()


def tokenize_dataset(texts: list[str], labels: list[int], tokenizer):
    """Convert texts to HuggingFace Dataset with tokenized inputs."""
    df = pd.DataFrame({"text": texts, "label": labels})
    hf_dataset = Dataset.from_pandas(df)

    def tokenize(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=config.MAX_TOKEN_LENGTH,
            padding=False,  # DataCollatorWithPadding handles dynamic padding
        )

    return hf_dataset.map(tokenize, batched=True, remove_columns=["text"])


def compute_metrics(eval_pred):
    """Compute accuracy, precision, recall, and F1 for evaluation."""
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, predictions),
        "f1": f1_score(labels, predictions, average="binary"),
        "precision": precision_score(labels, predictions, average="binary"),
        "recall": recall_score(labels, predictions, average="binary"),
    }


class WeightedTrainer(Trainer):
    """Custom Trainer that applies class weights to handle imbalanced datasets."""

    def __init__(self, class_weights, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        weight_tensor = torch.tensor(self.class_weights, dtype=torch.float).to(logits.device)
        loss_fn = torch.nn.CrossEntropyLoss(weight=weight_tensor)
        loss = loss_fn(logits, labels)
        return (loss, outputs) if return_outputs else loss


def train():
    # ── 1. Load Data ───────────────────────────────────────────────────────
    texts, labels = load_data()

    # ── 2. Train/Test Split ────────────────────────────────────────────────
    print(f"\n[2/5] Splitting dataset: {int((1-config.TEST_SIZE)*100)}% train / {int(config.TEST_SIZE*100)}% test")
    train_texts, test_texts, train_labels, test_labels = train_test_split(
        texts, labels,
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_SEED,
        stratify=labels,  # preserve class ratio in both splits
    )
    print(f"      Train: {len(train_texts)} | Test: {len(test_texts)}")

    # ── 3. Load Tokenizer & Tokenize ───────────────────────────────────────
    print(f"\n[3/5] Loading tokenizer: {config.PRETRAINED_MODEL}")
    tokenizer = AutoTokenizer.from_pretrained(config.PRETRAINED_MODEL)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    train_dataset = tokenize_dataset(train_texts, train_labels, tokenizer)
    test_dataset = tokenize_dataset(test_texts, test_labels, tokenizer)

    # ── 4. Load Model ──────────────────────────────────────────────────────
    print(f"\n[4/5] Loading model: {config.PRETRAINED_MODEL}")
    model = AutoModelForSequenceClassification.from_pretrained(
        config.PRETRAINED_MODEL,
        num_labels=config.NUM_LABELS,
        id2label={0: "genuine", 1: "fraudulent"},
        label2id={"genuine": 0, "fraudulent": 1},
    )

    # ── 5. Train ───────────────────────────────────────────────────────────
    print(f"\n[5/5] Starting fine-tuning...")
    config.MODEL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    config.LOGS_DIR.mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=str(config.MODEL_OUTPUT_DIR),
        logging_dir=str(config.LOGS_DIR),
        num_train_epochs=config.NUM_EPOCHS,
        per_device_train_batch_size=config.BATCH_SIZE,
        per_device_eval_batch_size=config.BATCH_SIZE,
        learning_rate=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY,
        warmup_ratio=config.WARMUP_RATIO,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        seed=config.RANDOM_SEED,
        report_to="none",  # disable wandb/tensorboard unless you want it
    )

    if config.USE_CLASS_WEIGHTS:
        class_weights = compute_class_weight(
            class_weight="balanced",
            classes=np.array([0, 1]),
            y=train_labels,
        )
        print(f"      Class weights: genuine={class_weights[0]:.3f}, fraudulent={class_weights[1]:.3f}")
        trainer = WeightedTrainer(
            class_weights=class_weights,
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=test_dataset,
            tokenizer=tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
        )
    else:
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=test_dataset,
            tokenizer=tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
        )

    trainer.train()

    # ── Save final model ───────────────────────────────────────────────────
    trainer.save_model(str(config.MODEL_OUTPUT_DIR))
    tokenizer.save_pretrained(str(config.MODEL_OUTPUT_DIR))
    print(f"\nModel saved to: {config.MODEL_OUTPUT_DIR}")


if __name__ == "__main__":
    train()
