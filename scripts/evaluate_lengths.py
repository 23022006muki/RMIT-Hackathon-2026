"""Measure truncation effects on held-out data without retraining or test labels."""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys
import time

import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from protector.data import load_split, read_data, sha256, write_json
from protector.metrics import score
from protector.training import config_from_file, EncodedPrompts, infer_logits, make_loader, seed_everything


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--data", default="../data")
    parser.add_argument("--split", default="artifacts/audit/split.csv")
    parser.add_argument("--output", default="artifacts/length_ablation")
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    experiment = Path(args.experiment)
    checkpoint = experiment / "checkpoint"
    run = json.loads((experiment / "run.json").read_text(encoding="utf-8"))
    if "total_seconds" not in run:
        raise ValueError("Use a completed experiment")
    base_config = config_from_file(checkpoint / "protector_config.json")
    seed_everything(base_config.seed, args.threads)
    tr, _, _ = read_data(args.data)
    _, validation = load_split(tr, args.split)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True, trust_remote_code=False)
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint, local_files_only=True, trust_remote_code=False)
    device = torch.device("cpu")
    model.to(device)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    rows, details = [], []
    for length, truncation in ((256, "head_tail"), (256, "head"), (512, "head_tail")):
        started = time.time()
        config = replace(base_config, max_length=length, truncation=truncation)
        dataset = EncodedPrompts(validation, tokenizer, config)
        logits = infer_logits(model, make_loader(dataset, tokenizer, 8), device)
        metrics, _ = score(validation, logits)
        row = {"max_length": length, "truncation": truncation, "macro_f1": metrics["macro_f1"], "truncated_rows": dataset.truncated, "seconds": round(time.time() - started, 2)}
        for label, stat in metrics["class_report"].items():
            if isinstance(stat, dict) and "f1-score" in stat:
                row[label] = stat["f1-score"]
        rows.append(row)
        details.append({**row, "metrics": metrics})
        print(json.dumps(row), flush=True)
        write_json(output / "results.json", {"experiment": str(experiment.resolve()), "trained_max_length": base_config.max_length, "validation_rows": len(validation), "split_sha256": sha256(args.split), "device": "cpu", "dtype": "float32", "conditions": details, "caveat": "Inference-only ablation on the selected checkpoint and the same validation split. This does not establish performance after retraining at a different length. CPU FP32 versus original CUDA FP16 can cause small numerical differences. No test labels or manual relabeling are used."})
        pd.DataFrame(rows).to_csv(output / "summary.csv", index=False)


if __name__ == "__main__":
    main()
