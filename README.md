# Signal Commons — RMIT Hackathon 2026, Challenges 1 × 2

Signal Commons is a multilingual AI safety research commons. The Challenge 1 entry trains **one approved XLM-RoBERTa model on Kaggle** to distinguish benign prompts, jailbreaks and prompt injections in Portuguese, French, Italian and English. The Challenge 2 app applies that same Kaggle-trained checkpoint to a contributor's prompt, then uses TideCloak to self-encrypt the original in the browser. The server stores ciphertext and classification metadata; the public sees aggregate patterns, while each contributor can retrieve and decrypt their own text.

The trust constraint is that an app administrator cannot simply open the stored originals. TideCloak identity, role checks and distributed key operations govern access. The classifier necessarily sees plaintext during inference, so the prototype protects the stored research archive rather than hiding prompts from the classifier. This creates a shared view of multilingual attack patterns without centralizing the underlying examples. [Explore the working app and setup](tide-app/README.md).

## Challenge 1 — Multilingual Protector

**RMIT · SGS · Lục Vị Hoàng Đế** — report **RMIT_SGS_LucViHoangDe_C1**. The executed notebook retains its original submission-time name **UEL_SGS_LucViHoangDe_C1**; the report school was updated at the user's request after scoring.

Classify Portuguese, French, Italian and English prompts into `BENIGN`, `JAILBREAK`, or `PROMPT_INJECTION`. Optimize three-class macro F1 using a reproducible language × label stratified split with detected near duplicates grouped together.

## Mandatory competition rules

Organizer replies retrieved on 22 October 2026 require **all training inside Kaggle**, **exactly one approved model**, **no ensembles including multiple fine-tuned versions of the same model**, and **no external or synthetic training data**. See [requirements and official links](docs/competition.md) and [source hashes/reply IDs](docs/rules_evidence.json).

The official entry selects **FacebookAI/xlm-roberta-base**, approved Kaggle source `aisuko/xlm-roberta-base/transformers/default/1`. The notebook contains only configurations for that backbone. It compares ordinary cross entropy and square-root inverse-frequency class weights, checks a second seed when their validation scores differ by less than 0.005, selects one configuration, and retrains one final model on all 14,000 training rows. Predictions are never ensembled.

**Earlier local experiments are ineligible archives.** They were stopped after the organizer replies became accessible. Their checkpoint files and historical measurements remain on the machine for transparency; they cannot select or initialize the official entry. Production training commands now reject a local runtime. The [main report](reports/RESULTS.md) excludes these runs. [Historical report](reports/local_only/README.md).

## Execute on Kaggle

1. Import [the self-contained notebook](notebooks/UEL_SGS_LucViHoangDe_C1.ipynb) and set its exact title above.
2. Attach the `rmit-hackathon-2026` competition input and the approved Kaggle XLM-RoBERTa model version 1. Enable a GPU. Phone verification may be needed for Kaggle GPU access.
3. Prepare compatible dependencies if the notebook's version checks identify an incompatible image; use official PyPI/PyTorch packages. Then execute the complete pipeline from original pretrained weights.
4. The notebook audits data, creates its split on Kaggle, runs validation experiments, measures inference truncation effects, retrains the selected configuration on the full train set, and validates `artifacts/final/submission.csv`.
5. Save a completed private notebook version and download all outputs to this project. Keep checkpoints available locally. The notebook does not automatically submit; one CSV was subsequently submitted with the user's explicit authorization.

Official Kaggle run directories use the `artifacts/kaggle_` prefix to distinguish them from the historical local archives. Completion and checkpoint-origin metadata are required before selection, reporting or packaging. Input SHA256 checks ensure that only the supplied competition CSVs and original approved model source are used. Complete runtime provenance includes seed, versions, GPU, configuration, data/split hashes, timing and metrics.

## Local files and commands

Repository: `D:\RMIT Hackathon 2026\project`. Raw data: `D:\RMIT Hackathon 2026\data`. Python: `D:\RMIT Hackathon 2026\.venv`. Downloaded original model cache: `D:\RMIT Hackathon 2026\cache\kagglehub`. Checkpoints remain under each run's `checkpoint` directory, excluded from Git.

Run these commands from the repository root for local audit, software checks and source preparation:

```powershell
$py = '..\.venv\Scripts\python.exe'
& $py -m protector.cli audit --data ../data
& $py scripts/build_notebook.py
& $py -m unittest discover -s tests -v
```

The small random-model unit fixture tests software mechanics without competition prompts, labels or pretrained weights. It is not an entry model or synthetic competition training data.

After downloading the **completed Kaggle** output tree into `artifacts`, generate local reports and a convenient bundle:

```powershell
& $py -m protector.cli validate --data ../data --submission artifacts/final/submission.csv
& $py scripts/make_report.py
& $py scripts/package_deliverables.py
```

The package includes source, notebook, reports and actual predictions. Large model files and original data remain at paths recorded in its manifest. Do not package or present a local training archive as the entry.

## Validation and controls

- Strict checks: 14,000 training rows, 6,000 test rows, known schemas, no empty fields or duplicate IDs, and known labels/languages.
- Three-class macro F1, class precision/recall/F1/support, confusion matrix, per-language and character-length diagnostics.
- Approximate MinHash retrieval with verified word-trigram Jaccard pairs; groups stay together across the validation boundary. This cannot prove absence of all semantic templates.
- Input text is preserved apart from tokenizer truncation. Head/tail truncation keeps both ends after reserving model special tokens. Inference-only 256/512-token diagnostics do not claim the effect of retraining at another length.
- Final inference uses one model. Validation scores belong to the selection model; full-data model hidden-test performance remains unknown until scoring.
- CSV checks: exactly `id,label`, 6,000 rows, unique test IDs in sample order, no missing values or invalid labels.
- Dataset prompts are untrusted strings and never executed. No manual test relabeling, external data, translation model or pseudo-labeling.

## Publication and submission status

Deadline verified: **22 October 2026 at 18:00 Asia/Saigon** (11:00 UTC). Maximum team size four. API says six daily submissions and Challenge text says six attempts. One explicitly authorized CSV submission completed successfully with **public macro F1 0.94224**. See [submission evidence](artifacts/submission_status.json) and [actual validation report](reports/RESULTS.md). The final private leaderboard score and overall RMIT challenge weights are unknown.

The public GitHub repository contains code and aggregate artifacts. ID-level competition data/predictions and weights remain local. An automatic approval review required explicit authorization before publishing those row-level files. Foundational Rules also require public code to be shared through the competition Kaggle forum/notebooks under an OSI-approved license permitting commercial use. The user authorized MIT for first-party code and publication of the [Kaggle source-link post](docs/kaggle_discussion_draft.md). MIT is applied and the [Kaggle sharing post](https://www.kaggle.com/competitions/rmit-hackathon-2026/discussion/751585) was published and verified. A private notebook run does not complete this public-sharing obligation.

Competition data has **CC BY-NC-SA 4.0**, separate from source-code licensing. No rules from 2025 are assumed. The authorized source-link Discussion post is public. The training notebook remains private. The completed Kaggle output archive, final checkpoint, candidate checkpoints and actual CSV are retained locally; row-level outputs and private execution links are excluded from Git.
