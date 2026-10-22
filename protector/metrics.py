import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, f1_score

from .constants import LABELS, LABEL2ID, LANGUAGES


def score(frame, logits, bias=None):
    y = frame.label.map(LABEL2ID).to_numpy()
    pred = (np.asarray(logits) + (np.zeros(3) if bias is None else np.asarray(bias))).argmax(axis=1)
    result = {
        "macro_f1": float(f1_score(y, pred, labels=[0, 1, 2], average="macro", zero_division=0)),
        "class_report": classification_report(y, pred, labels=[0, 1, 2], target_names=LABELS, output_dict=True, zero_division=0),
        "confusion_matrix": confusion_matrix(y, pred, labels=[0, 1, 2]).tolist(),
        "confusion_matrix_labels": LABELS,
        "by_language": {},
        "by_length_characters": {},
    }
    for language in LANGUAGES:
        mask = frame.language.eq(language).to_numpy()
        if mask.any():
            result["by_language"][language] = {"n": int(mask.sum()), "macro_f1": float(f1_score(y[mask], pred[mask], labels=[0, 1, 2], average="macro", zero_division=0))}
    lengths = frame.text.str.len().to_numpy()
    for name, mask in (("0-299", lengths < 300), ("300-999", (lengths >= 300) & (lengths < 1000)), ("1000-1999", (lengths >= 1000) & (lengths < 2000)), ("2000+", lengths >= 2000)):
        if mask.any():
            result["by_length_characters"][name] = {"n": int(mask.sum()), "macro_f1": float(f1_score(y[mask], pred[mask], labels=[0, 1, 2], average="macro", zero_division=0))}
    return result, pred


def export_predictions(frame, logits, path, bias=None):
    _, pred = score(frame, logits, bias)
    result = frame[["id", "language", "label"]].copy()
    result["prediction"] = [LABELS[i] for i in pred]
    for i, label in enumerate(LABELS):
        result[f"logit_{label}"] = np.asarray(logits)[:, i]
    result["characters"] = frame.text.str.len().to_numpy()
    result.to_csv(path, index=False)
    return result
