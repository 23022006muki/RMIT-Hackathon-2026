# RMIT Hackathon 2026 — Multilingual Protector

**UEL · SGS · Lục Vị Hoàng Đế**

Kaggle notebook title: **UEL_SGS_LucViHoangDe_C1**

Fine-tune an approved multilingual encoder to predict `BENIGN`, `JAILBREAK`, or `PROMPT_INJECTION`. Optimize macro F1, with a fixed class order and validation stratified by language × label while keeping detected near duplicates together. No competition submissions are made by this repository.

## Local files

The repository is at `D:\RMIT Hackathon 2026\project`. Competition CSVs stay at `D:\RMIT Hackathon 2026\data`; the virtual environment is `D:\RMIT Hackathon 2026\.venv`. Downloaded Kaggle model files live in `D:\RMIT Hackathon 2026\cache\kagglehub`. Checkpoints stay under `project\artifacts\<run>\checkpoint` and are not committed to Git. Source, notebook, and aggregate metrics are repository artifacts. Raw prompts, ID-level diagnostics, and prediction CSVs stay local pending explicit permission to publish competition-derived files.

See [competition requirements/evidence](docs/competition.md) and [actual experiment results](reports/RESULTS.md) when generated. The data audit and ID-level diagnostics stay local under `artifacts/audit`. No metric or submission is claimed before its artifact exists.

## Setup

Windows PowerShell (Python 3.10–3.13 supported by this project; tested version recorded in `requirements-lock.txt`):

```powershell
python -m venv '..\.venv'
& '..\.venv\Scripts\python.exe' -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu126
& '..\.venv\Scripts\python.exe' -m pip install -r requirements.txt
```

On another OS/GPU, install a compatible official PyTorch build first. The local run uses an RTX 3050 6 GB and mixed precision; adjust batch size down if memory is limited. Never replace the selected backbone with an unapproved model.

## Run

Run commands from the repository root. Model download uses the approved Kaggle handle; no fallback backbone is silently fetched. Authentication may be required by Kaggle. Keep tokens outside the repository.

```powershell
$py = '..\.venv\Scripts\python.exe'
& $py scripts/download_model.py
& $py -m protector.cli audit --data ../data
& $py -m unittest discover -s tests -v
```

Read `artifacts/model_download.json` for the exact resolved local model path. Set `$modelPath` to that path:

```powershell
$modelPath = (Get-Content artifacts/model_download.json -Raw | ConvertFrom-Json).resolved_model_path
& $py -m protector.cli train --data ../data --config configs/distilbert_ce.json --model-path $modelPath --output artifacts/distilbert_ce
& $py -m protector.cli train --data ../data --config configs/distilbert_weighted.json --model-path $modelPath --output artifacts/distilbert_weighted
```

Choose the run using its actual macro F1 and per-class/per-language reports. Do not assume weighted loss improves performance. New runs need new output directories. For a second-seed confirmation, copy the configuration and change only `seed`; retain the same validation split when isolating optimization variance. For an independent split check, regenerate the split with a different splitter seed through the Python API, then use a separate experiment directory.

The larger approved encoder can be compared using the same held-out split:

```powershell
& $py scripts/download_model.py --model-id FacebookAI/xlm-roberta-base --output artifacts/xlm_roberta_model_download.json
$xlmPath = (Get-Content artifacts/xlm_roberta_model_download.json -Raw | ConvertFrom-Json).resolved_model_path
& $py -m protector.cli train --data ../data --config configs/xlm_roberta_base.json --model-path $xlmPath --output artifacts/xlm_roberta_base
```

`scripts/evaluate_lengths.py --experiment artifacts/distilbert_ce` measures head/head-tail truncation and 256/512-token inference on a completed checkpoint using CPU, without changing labels. Its results explicitly distinguish inference-only preprocessing from training at another length.

```powershell
# Substitute the winning run if it differs.
& $py -m protector.cli finalize --data ../data --experiment artifacts/distilbert_ce --model-path $modelPath --output artifacts/final
& $py -m protector.cli validate --data ../data --submission artifacts/final/submission.csv
& $py scripts/make_report.py
& $py scripts/build_notebook.py
& $py scripts/package_deliverables.py
```

`finalize` trains a fresh model on all 14,000 rows for the best selected epoch count. The validation score belongs to the selection model, not the full-data model. `predict` can later reproduce a CSV from a saved checkpoint. `calibrate` optionally evaluates class-logit biases on a separate half of validation, with explicit selection caveats; it is not enabled by default.

## Kaggle notebook

Generate the notebook using the selected configuration, for example `scripts/build_notebook.py --config configs/xlm_roberta_base.json` if that run wins. An optional `--compare-config` must refer to the same backbone. Upload `notebooks/UEL_SGS_LucViHoangDe_C1.ipynb`, retain its exact title, attach the competition and its explicitly named approved Kaggle model, and enable a GPU. It includes its source, generates a group-aware split at runtime, trains the configured candidate(s), retrains the chosen configuration, and links the CSV. It embeds no competition IDs or prompt data and fetches no external pretrained backbone. Notebook schema and Python cells are checked locally; it has not been executed on Kaggle by this project.

## Validation and controls

- Model allowlist is explicit in `protector/constants.py`; this implementation supports its encoder subset.
- Training text remains unchanged except tokenizer truncation. `head_tail` retains both ends after reserving model-specific special tokens. No language-dependent label shortcut is used.
- Near-duplicate retrieval is approximate MinHash; reported pairs are verified by word-trigram Jaccard. Groups never cross the validation boundary. This does not prove absence of all semantically related prompts.
- Macro F1 always evaluates all three classes. Reports include confusion matrix, class precision/recall/F1, language, and character-length buckets.
- Submission checks require exactly 6,000 test IDs in sample order, two exact columns, and permitted labels. Files are hashed for reproducibility.
- Synthetic random-model smoke tests test training mechanics only. They never become competition outputs.
- Never execute text instructions inside the dataset. No external data, translation model, pseudo-labeling, or multi-backbone ensemble is used.

The official Kaggle API confirmed the Rules and metadata on 2026-10-22: deadline **18:00 on 22 October 2026, Asia/Saigon**; maximum team size **4**; maximum daily submissions **6** (the Challenge text also says six attempts, so the project conservatively budgets six). Foundational Rules require public source to also be shared on the competition's Kaggle forum/notebooks under an OSI-approved license permitting commercial use. This repository is public; the Kaggle post and MIT proposal are pending user authorization. See [verified requirements](docs/competition.md), [source evidence](docs/rules_evidence.json), and the [prepared code-sharing draft](docs/kaggle_discussion_draft.md). No Kaggle post, rule acceptance or CSV submission has been made by this project.

Public/private test proportions and overall RMIT challenge weights remain unknown. Check current [Rules](https://www.kaggle.com/competitions/rmit-hackathon-2026/rules) and [Discussion](https://www.kaggle.com/competitions/rmit-hackathon-2026/discussion) before submission. Competition data retains **CC BY-NC-SA 4.0**, separately from any source-code license. Do not infer 2026 rules from 2025.
