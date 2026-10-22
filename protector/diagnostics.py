"""Kaggle-only validation truncation diagnostics for one selected checkpoint."""
from dataclasses import replace
import gc
import json
from pathlib import Path
import time

import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from .data import load_split, read_data, sha256, write_json
from .metrics import score
from .runtime import require_kaggle_runtime, verify_competition_input
from .training import config_from_file, EncodedPrompts, infer_logits, make_loader, seed_everything


def evaluate_lengths(experiment, data, split, output):
    execution = require_kaggle_runtime()
    verify_competition_input(data)
    experiment, output = Path(experiment), Path(output)
    run = json.loads((experiment / "run.json").read_text(encoding="utf-8"))
    if "total_seconds" not in run or run.get("execution", {}).get("platform") != "kaggle":
        raise ValueError("Use a completed Kaggle experiment")
    checkpoint = experiment / "checkpoint"
    config = config_from_file(checkpoint / "protector_config.json")
    require_kaggle_runtime(config.model_id)
    seed_everything(config.seed, config.num_threads)
    train, _, _ = read_data(data)
    _, validation = load_split(train, split)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True, trust_remote_code=False)
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint, local_files_only=True, trust_remote_code=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    output.mkdir(parents=True, exist_ok=True)
    rows, details = [], []
    for length, truncation in ((256, "head_tail"), (256, "head"), (512, "head_tail")):
        start = time.time()
        dataset = EncodedPrompts(validation, tokenizer, replace(config, max_length=length, truncation=truncation))
        logits = infer_logits(model, make_loader(dataset, tokenizer, 8), device)
        metrics, _ = score(validation, logits)
        row = {"max_length": length, "truncation": truncation, "macro_f1": metrics["macro_f1"], "truncated_rows": dataset.truncated, "seconds": round(time.time()-start, 2)}
        row.update({label: stat["f1-score"] for label, stat in metrics["class_report"].items() if isinstance(stat, dict) and "f1-score" in stat})
        rows.append(row)
        details.append({**row, "metrics": metrics})
        write_json(output / "results.json", {"experiment": str(experiment.resolve()), "execution": execution, "trained_max_length": config.max_length, "validation_rows": len(validation), "split_sha256": sha256(split), "device": str(device), "conditions": details, "caveat": "Inference-only diagnostics on one Kaggle checkpoint and the same validation split. Final training/inference retain the original configuration; this is not evidence for retraining at another length."})
        pd.DataFrame(rows).to_csv(output / "summary.csv", index=False)
        print(json.dumps(row), flush=True)
    del model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return details
