import gc
import json
import math
import os
import random
import time
from dataclasses import asdict, dataclass, fields
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer
from transformers import DataCollatorWithPadding, get_linear_schedule_with_warmup

from .constants import ENCODERS, LABEL2ID, LABELS
from .data import load_split, read_data, sha256, validate_submission, write_json
from .metrics import export_predictions, score

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")


@dataclass
class TrainConfig:
    model_id: str = "distilbert/distilbert-base-multilingual-cased"
    max_length: int = 256
    truncation: str = "head_tail"
    epochs: int = 3
    batch_size: int = 8
    eval_batch_size: int = 16
    accumulation_steps: int = 4
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.06
    weight_power: float = 0.0
    seed: int = 42
    gradient_checkpointing: bool = False
    patience: int = 2
    num_threads: int = 4
    optimizer_foreach: bool | None = None

    def __post_init__(self):
        if self.model_id not in ENCODERS:
            raise ValueError("This pipeline supports only the approved encoder backbones")
        if self.truncation not in {"head", "head_tail"}:
            raise ValueError("truncation must be head or head_tail")
        if min(self.epochs, self.batch_size, self.eval_batch_size, self.accumulation_steps) < 1:
            raise ValueError("Training sizes must be positive")
        if not 16 <= self.max_length <= 512 or self.weight_power < 0:
            raise ValueError("Invalid max_length or weight_power")


def config_from_file(path):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    unexpected = set(data) - {f.name for f in fields(TrainConfig)}
    if unexpected:
        raise ValueError(f"Unknown training fields: {unexpected}")
    return TrainConfig(**data)


def seed_everything(seed, threads=4):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(threads)
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)


class EncodedPrompts(Dataset):
    def __init__(self, frame, tokenizer, config):
        self.items = []
        self.truncated = 0
        self.total = len(frame)
        budget = config.max_length - tokenizer.num_special_tokens_to_add(pair=False)
        if budget < 1:
            raise ValueError("max_length cannot hold model special tokens")
        raw = tokenizer(frame.text.tolist(), add_special_tokens=False, truncation=False, verbose=False)["input_ids"]
        for i, ids in enumerate(raw):
            if len(ids) > budget:
                self.truncated += 1
                if config.truncation == "head_tail":
                    left = (budget + 1) // 2
                    ids = ids[:left] + ids[-(budget - left):]
                else:
                    ids = ids[:budget]
            # prepare_for_model handles model-specific [CLS]/[SEP]/<s> tokens.
            item = tokenizer.prepare_for_model(ids, add_special_tokens=True, return_attention_mask=True, truncation=False, verbose=False)
            if "label" in frame:
                item["labels"] = LABEL2ID[frame.iloc[i].label]
            self.items.append(item)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        return self.items[index]


def make_loader(dataset, tokenizer, batch_size, shuffle=False, seed=42):
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0,
                      pin_memory=torch.cuda.is_available(), generator=generator,
                      collate_fn=DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8, return_tensors="pt"))


def infer_logits(model, loader, device):
    model.eval()
    predictions = []
    with torch.inference_mode():
        for batch in loader:
            inputs = {k: v.to(device, non_blocking=True) for k, v in batch.items() if k != "labels"}
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                output = model(**inputs)
            predictions.append(output.logits.float().cpu().numpy())
    return np.concatenate(predictions)


def run_training(config, model_path, train_frame, validation_frame, output, provenance):
    output = Path(output)
    if (output / "history.json").exists():
        raise ValueError(f"Run already exists: {output}; use a new run directory")
    output.mkdir(parents=True, exist_ok=True)
    model_path = Path(model_path).resolve()
    if not (model_path / "config.json").is_file():
        raise FileNotFoundError("model-path must be a local approved model with config.json")
    seed_everything(config.seed, config.num_threads)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    start = time.time()
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, use_fast=True, trust_remote_code=False)
    model_config = AutoConfig.from_pretrained(model_path, local_files_only=True, trust_remote_code=False)
    model_config.num_labels = 3
    model_config.label2id = LABEL2ID
    model_config.id2label = dict(enumerate(LABELS))
    # Existing 2-class safety heads are deliberately replaced by a 3-class head.
    model = AutoModelForSequenceClassification.from_pretrained(model_path, config=model_config, ignore_mismatched_sizes=True, local_files_only=True, trust_remote_code=False)
    if config.gradient_checkpointing:
        model.gradient_checkpointing_enable()
    model.to(device)
    train_ds = EncodedPrompts(train_frame, tokenizer, config)
    train_loader = make_loader(train_ds, tokenizer, config.batch_size, True, config.seed)
    val_loader = None
    if validation_frame is not None:
        val_ds = EncodedPrompts(validation_frame, tokenizer, config)
        val_loader = make_loader(val_ds, tokenizer, config.eval_batch_size)
    counts = np.bincount(train_frame.label.map(LABEL2ID), minlength=3)
    if np.any(counts == 0):
        raise ValueError("All three classes must appear in training")
    weights = (len(train_frame) / (3.0 * counts)) ** config.weight_power
    weights /= weights.mean()
    class_weights = torch.tensor(weights, dtype=torch.float32, device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay, foreach=config.optimizer_foreach)
    updates_per_epoch = math.ceil(len(train_loader) / config.accumulation_steps)
    total_updates = updates_per_epoch * config.epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, int(total_updates * config.warmup_ratio), total_updates)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    runtime = {
        "config": asdict(config), "device": str(device), "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "train_rows": len(train_frame), "validation_rows": len(validation_frame) if validation_frame is not None else 0,
        "train_truncated": train_ds.truncated,
        "validation_truncated": val_ds.truncated if val_loader is not None else 0,
        "class_weights": weights.tolist(), "provenance": provenance,
        "pretrained_config_sha256": sha256(model_path / "config.json"),
    }
    write_json(output / "run.json", runtime)
    best, stale, history = -1.0, 0, []
    print(f"Training on {device}: {len(train_frame)} rows, {len(train_loader)} batches/epoch", flush=True)
    for epoch in range(1, config.epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        running_loss, seen = 0.0, 0
        epoch_start = time.time()
        for step, batch in enumerate(train_loader, 1):
            labels = batch.pop("labels").to(device, non_blocking=True)
            inputs = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            group_start = ((step - 1) // config.accumulation_steps) * config.accumulation_steps
            # The final partial accumulation group must have the correct scale.
            group_size = min(config.accumulation_steps, len(train_loader) - group_start)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                logits = model(**inputs).logits
                loss = torch.nn.functional.cross_entropy(logits.float(), labels, weight=class_weights)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite loss at epoch {epoch}, batch {step}")
            scaler.scale(loss / group_size).backward()
            running_loss += loss.item() * len(labels)
            seen += len(labels)
            if step % config.accumulation_steps == 0 or step == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                previous_scale = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                if scaler.get_scale() >= previous_scale:
                    scheduler.step()
                optimizer.zero_grad(set_to_none=True)
            if step == 1 or step % 100 == 0:
                print(json.dumps({"epoch": epoch, "batch": step, "batches": len(train_loader), "loss": round(running_loss / seen, 5), "elapsed_seconds": round(time.time() - epoch_start)}), flush=True)
        row = {"epoch": epoch, "train_loss": running_loss / seen, "seconds": round(time.time() - epoch_start, 2)}
        metric = None
        if val_loader is not None:
            val_logits = infer_logits(model, val_loader, device)
            metric, _ = score(validation_frame, val_logits)
            row["macro_f1"] = metric["macro_f1"]
            improved = metric["macro_f1"] > best + 1e-8
        else:
            improved = epoch == config.epochs
        history.append(row)
        write_json(output / "history.json", history)
        print(json.dumps(row), flush=True)
        if improved:
            model.save_pretrained(output / "checkpoint", safe_serialization=True)
            tokenizer.save_pretrained(output / "checkpoint")
            write_json(output / "checkpoint" / "protector_config.json", asdict(config))
            write_json(output / "best_epoch.json", {"epoch": epoch, "macro_f1": metric["macro_f1"] if metric else None})
            if metric:
                best = metric["macro_f1"]
                write_json(output / "metrics.json", metric)
                np.save(output / "validation_logits.npy", val_logits)
                export_predictions(validation_frame, val_logits, output / "validation_predictions.csv")
            stale = 0
        else:
            stale += 1
        if validation_frame is not None and stale >= config.patience:
            break
    runtime["total_seconds"] = round(time.time() - start, 2)
    runtime["peak_gpu_allocated_gb"] = round(torch.cuda.max_memory_allocated() / 1024**3, 3) if device.type == "cuda" else 0
    write_json(output / "run.json", runtime)
    del model, optimizer, scheduler, scaler
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return output


def train_experiment(data_dir, split_path, config_path, model_path, output):
    config = config_from_file(config_path)
    tr, _, _ = read_data(data_dir)
    train, val = load_split(tr, split_path)
    provenance = {"model_id": config.model_id, "model_source": str(Path(model_path).resolve()), "config_file_sha256": sha256(config_path), "train_file_sha256": sha256(Path(data_dir) / "train.csv"), "split_sha256": sha256(split_path)}
    source = Path(model_path) / "model_provenance.json"
    if source.exists():
        provenance["download"] = json.loads(source.read_text(encoding="utf-8"))
        if provenance["download"]["model_id"] != config.model_id:
            raise ValueError("Model provenance does not match configured approved ID")
    return run_training(config, model_path, train, val, output, provenance)


def predict(data_dir, checkpoint, output, bias_path=None):
    checkpoint, output = Path(checkpoint), Path(output)
    tr, te, sample = read_data(data_dir)
    config = config_from_file(checkpoint / "protector_config.json")
    seed_everything(config.seed, config.num_threads)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True, trust_remote_code=False)
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint, local_files_only=True, trust_remote_code=False)
    if model.config.id2label != dict(enumerate(LABELS)):
        raise ValueError("Checkpoint class order does not match submission labels")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    ds = EncodedPrompts(te, tokenizer, config)
    logits = infer_logits(model, make_loader(ds, tokenizer, config.eval_batch_size), device)
    bias = np.zeros(3)
    if bias_path:
        calibration = json.loads(Path(bias_path).read_text(encoding="utf-8"))
        bias = np.asarray(calibration["selected_bias"])
    labels = [LABELS[i] for i in (logits + bias).argmax(1)]
    mapping = dict(zip(te.id, labels, strict=True))
    sample["label"] = sample.id.map(mapping)
    output.mkdir(parents=True, exist_ok=True)
    sample.to_csv(output / "submission.csv", index=False)
    np.save(output / "test_logits.npy", logits)
    validation = validate_submission(output / "submission.csv", data_dir)
    validation["checkpoint"] = str(checkpoint.resolve())
    validation["truncated_test_rows"] = ds.truncated
    validation["bias"] = bias.tolist()
    write_json(output / "submission_validation.json", validation)
    print(json.dumps(validation, indent=2), flush=True)
    del model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return validation


def finalize(data_dir, experiment, model_path, output, bias_path=None):
    experiment, output = Path(experiment), Path(output)
    run = json.loads((experiment / "run.json").read_text(encoding="utf-8"))
    config = TrainConfig(**run["config"])
    best = json.loads((experiment / "best_epoch.json").read_text(encoding="utf-8"))
    config.epochs = best["epoch"]
    tr, _, _ = read_data(data_dir)
    provenance = {"selected_experiment": str(experiment.resolve()), "validation_macro_f1": best["macro_f1"], "final_epochs": config.epochs, "train_file_sha256": sha256(Path(data_dir) / "train.csv"), "pretrained_provenance": run["provenance"], "note": "Final model trained from approved pretrained initialization on all 14,000 train rows. Reported validation metric belongs to the selection model, not this full-data model."}
    run_training(config, model_path, tr, None, output, provenance)
    return predict(data_dir, output / "checkpoint", output, bias_path)
