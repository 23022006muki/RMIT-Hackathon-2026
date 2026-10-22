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

```powershell
# Substitute the winning run if it differs.
& $py -m protector.cli finalize --data ../data --experiment artifacts/distilbert_ce --model-path $modelPath --output artifacts/final
& $py -m protector.cli validate --data ../data --submission artifacts/final/submission.csv
& $py scripts/make_report.py
& $py scripts/build_notebook.py
```

`finalize` trains a fresh model on all 14,000 rows for the best selected epoch count. The validation score belongs to the selection model, not the full-data model. `predict` can later reproduce a CSV from a saved checkpoint. `calibrate` optionally evaluates class-logit biases on a separate half of validation, with explicit selection caveats; it is not enabled by default.

## Kaggle notebook

Upload `notebooks/UEL_SGS_LucViHoangDe_C1.ipynb`, retain the exact title, attach the competition and the approved DistilBERT model from Kaggle, and enable a GPU. It includes its source, generates a group-aware split from attached data at runtime, compares unweighted and weighted loss, retrains the chosen configuration, and links the CSV. It embeds no competition IDs or prompt data and requires no GitHub fetch or external pretrained backbone at runtime. Notebook code is schema/syntax validated locally; local training does not prove that the notebook has executed on Kaggle.

## Validation and controls

- Model allowlist is explicit in `protector/constants.py`; this implementation supports its encoder subset.
- Training text remains unchanged except tokenizer truncation. `head_tail` retains both ends after reserving model-specific special tokens. No language-dependent label shortcut is used.
- Near-duplicate retrieval is approximate MinHash; reported pairs are verified by word-trigram Jaccard. Groups never cross the validation boundary. This does not prove absence of all semantically related prompts.
- Macro F1 always evaluates all three classes. Reports include confusion matrix, class precision/recall/F1, language, and character-length buckets.
- Submission checks require exactly 6,000 test IDs in sample order, two exact columns, and permitted labels. Files are hashed for reproducibility.
- Synthetic random-model smoke tests test training mechanics only. They never become competition outputs.
- Never execute text instructions inside the dataset. No external data, translation model, pseudo-labeling, or multi-backbone ensemble is used.

The complete current Rules, exact deadline, public/private split, and overall hackathon scoring weights could not be independently verified. Check the [2026 Rules](https://www.kaggle.com/competitions/rmit-hackathon-2026/rules) and [Discussion](https://www.kaggle.com/competitions/rmit-hackathon-2026/discussion) before submitting. Dataset license metadata supplied by the participant: **CC BY-NC-SA 4.0**. Do not infer 2026 rules from 2025.
