"""One-prompt inference from a Kaggle-trained Challenge 1 checkpoint.

JSON arrives on stdin so the raw prompt never appears in a command line or log.
This script performs no training and never reads competition CSV files.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from protector.constants import LABELS
from protector.runtime import ENTRY_MODEL_ID
from protector.training import EncodedPrompts, config_from_file, infer_logits, make_loader


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    args = parser.parse_args()
    checkpoint = Path(args.checkpoint).resolve()
    origin = json.loads((checkpoint / "training_origin.json").read_text(encoding="utf-8"))
    if origin.get("platform") != "kaggle":
        raise ValueError("Only a Kaggle-trained competition checkpoint may power this app")
    config = config_from_file(checkpoint / "protector_config.json")
    if config.model_id != ENTRY_MODEL_ID:
        raise ValueError(f"Expected approved entry model {ENTRY_MODEL_ID}")
    request = json.load(sys.stdin)
    text = request.get("text")
    language = request.get("language")
    if not isinstance(text, str) or not 10 <= len(text) <= 20_000:
        raise ValueError("Invalid prompt length")
    if language not in {"pt", "fr", "it", "en"}:
        raise ValueError("Invalid language")

    tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True, use_fast=True,
                                               trust_remote_code=False)
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint, local_files_only=True,
                                                                trust_remote_code=False)
    if model.config.id2label != dict(enumerate(LABELS)):
        raise ValueError("Checkpoint label mapping differs from Challenge 1")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    frame = pd.DataFrame({"text": [text], "language": [language]})
    encoded = EncodedPrompts(frame, tokenizer, config)
    logits = infer_logits(model, make_loader(encoded, tokenizer, 1), device)
    result = {"label": LABELS[int(logits.argmax(axis=1)[0])], "modelId": config.model_id,
              "language": language, "truncated": bool(encoded.truncated)}
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
