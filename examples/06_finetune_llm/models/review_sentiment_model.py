"""A fine-tuned DistilBERT — the student.

66 million parameters, fine-tuned on CPU in a few minutes, and it is *yours*: the
weights sit on a volume, inference costs nothing per call, and no endpoint can
rate-limit you or be deprecated out from under you.

It implements the same UbunyeModel contract as the sklearn model in example 04,
because it is the same kind of thing — something that learns from data and makes
predictions. That a transformer is doing the learning changes nothing about how it
must be governed: it still gets split, judged on data it never saw, gated, and
registered, or it does not ship.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from ubunye.models.base import UbunyeModel

# The open-source model we fine-tune. Small enough for CPU, good enough for
# sentiment. Swap it for another HuggingFace checkpoint and nothing else changes.
BASE_MODEL = "distilbert-base-uncased"

LABELS = ["negative", "neutral", "positive"]
LABEL_TO_ID = {label: i for i, label in enumerate(LABELS)}

MAX_LENGTH = 256
EPOCHS = 3
BATCH_SIZE = 8
LEARNING_RATE = 5e-5


class ReviewSentimentModel(UbunyeModel):
    """DistilBERT fine-tuned to reproduce the teacher's labels."""

    def __init__(self) -> None:
        self._model = None
        self._tokenizer = None

    # -- learn ---------------------------------------------------------------

    def train(self, df: Any) -> Dict[str, Any]:
        """Fine-tune on the teacher's labels. CPU, a few minutes."""
        import torch
        from torch.utils.data import DataLoader, TensorDataset
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        pdf = _to_pandas(df)
        torch.manual_seed(42)  # a training run you cannot reproduce is an anecdote

        self._tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
        self._model = AutoModelForSequenceClassification.from_pretrained(
            BASE_MODEL, num_labels=len(LABELS)
        )

        dataset = self._encode(pdf)
        loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
        optimiser = torch.optim.AdamW(self._model.parameters(), lr=LEARNING_RATE)

        self._model.train()
        losses: List[float] = []

        for epoch in range(EPOCHS):
            epoch_losses = []
            for input_ids, attention_mask, labels in loader:
                optimiser.zero_grad()
                out = self._model(
                    input_ids=input_ids, attention_mask=attention_mask, labels=labels
                )
                out.loss.backward()
                optimiser.step()
                epoch_losses.append(float(out.loss.item()))
            losses.append(float(np.mean(epoch_losses)))

        preds = self._predict_ids(pdf["review_text"].tolist())
        actual = pdf["label"].map(LABEL_TO_ID).to_numpy()

        return {
            "train_accuracy": float((preds == actual).mean()),
            "train_loss": losses[-1],
            "train_rows": float(len(pdf)),
            "epochs": float(EPOCHS),
        }

    # -- score honestly ------------------------------------------------------

    def validate(self, df: Any) -> Dict[str, Any]:
        """Agreement with the teacher on reviews the student never saw.

        This is the number that decides whether the distillation worked: if the
        student cannot reproduce the teacher on held-out data, you have not
        compressed the teacher — you have just made something cheaper and wrong.
        """
        pdf = _to_pandas(df)
        preds = self._predict_ids(pdf["review_text"].tolist())
        actual = pdf["label"].map(LABEL_TO_ID).to_numpy()

        metrics = {
            "test_accuracy": float((preds == actual).mean()),
            "test_rows": float(len(pdf)),
        }
        # Per-class recall, because a model that predicts the majority class for
        # everything can still post a respectable accuracy and be useless.
        for label, idx in LABEL_TO_ID.items():
            mask = actual == idx
            if mask.sum():
                metrics[f"test_recall_{label}"] = float((preds[mask] == idx).mean())
        return metrics

    # -- use -----------------------------------------------------------------

    def predict(self, df: Any) -> Any:
        pdf = _to_pandas(df).copy()
        ids = self._predict_ids(pdf["review_text"].tolist())
        pdf["predicted_label"] = [LABELS[i] for i in ids]
        return pdf

    def _predict_ids(self, texts: List[str]) -> np.ndarray:
        import torch

        self._model.eval()
        out_ids: List[int] = []

        with torch.no_grad():
            for start in range(0, len(texts), 16):
                batch = texts[start : start + 16]
                encoded = self._tokenizer(
                    batch,
                    truncation=True,
                    padding=True,
                    max_length=MAX_LENGTH,
                    return_tensors="pt",
                )
                logits = self._model(**encoded).logits
                out_ids.extend(torch.argmax(logits, dim=1).tolist())

        return np.array(out_ids)

    def _encode(self, pdf: pd.DataFrame):
        import torch
        from torch.utils.data import TensorDataset

        encoded = self._tokenizer(
            pdf["review_text"].tolist(),
            truncation=True,
            padding="max_length",
            max_length=MAX_LENGTH,
            return_tensors="pt",
        )
        labels = torch.tensor(pdf["label"].map(LABEL_TO_ID).tolist())
        return TensorDataset(encoded["input_ids"], encoded["attention_mask"], labels)

    # -- persist -------------------------------------------------------------

    def save(self, path: str) -> None:
        Path(path).mkdir(parents=True, exist_ok=True)
        self._model.save_pretrained(path)
        self._tokenizer.save_pretrained(path)

    @classmethod
    def load(cls, path: str) -> "ReviewSentimentModel":
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        model = cls()
        model._model = AutoModelForSequenceClassification.from_pretrained(path)
        model._tokenizer = AutoTokenizer.from_pretrained(path)
        return model

    def metadata(self) -> Dict[str, Any]:
        import transformers

        return {
            "library": "transformers",
            "library_version": transformers.__version__,
            "features": ["review_text"],
            "target": "label",
            "params": {
                "base_model": BASE_MODEL,
                "labels": LABELS,
                "epochs": EPOCHS,
                "batch_size": BATCH_SIZE,
                "learning_rate": LEARNING_RATE,
                "max_length": MAX_LENGTH,
            },
        }


def _to_pandas(df: Any) -> pd.DataFrame:
    if isinstance(df, pd.DataFrame):
        return df
    return df.toPandas()
