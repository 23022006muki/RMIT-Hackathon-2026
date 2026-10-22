# Competition requirements and evidence

Team: **Lục Vị Hoàng Đế**, university **UEL**, campus **SGS**. Notebook title: **UEL_SGS_LucViHoangDe_C1**.

Evidence checked on 2026-10-22 (Asia/Saigon). Overview and Data content were supplied by the participant in this chat. Direct web fetches could not read the competition, browser requests returned Kaggle rate limits, and direct HTML downloads returned only the JavaScript shell. Do not describe the full Rules as independently verified.

| Requirement | Evidence | Implementation |
|---|---|---|
| Three exact labels; multilingual pt/fr/it/en | Supplied 2026 Overview/Data; CSV schema | Fixed label mapping and input validation |
| 14,000 train; 6,000 test; id,text,language,label | Supplied Data; local CSV audit | Strict row/schema checks and file hashes |
| Macro-averaged F1 with equal class weight | Supplied Evaluation | Explicit three-class macro F1, per-class precision/recall/F1 |
| Fine-tune an approved pretrained backbone | Supplied Models table in Overview | Explicit allowlist; selected approved multilingual DistilBERT; Kaggle source provenance |
| Name notebook `<university>_<campus>_<team>_C1` | Supplied Overview | Title uses UEL_SGS_LucViHoangDe_C1 |
| State chosen model at notebook top | Supplied Overview | First markdown cell names backbone |
| Exactly id,label and all 6,000 test IDs | Supplied Evaluation; local sample | CSV validation including ID order and allowed labels |
| Six submission attempts | Supplied Overview | No code submits to Kaggle; submissions used by this project: 0 |
| First valid result earns full Challenge 1 marks | Supplied Overview | Not interpreted as a guarantee of high leaderboard rank |
| Dataset license CC BY-NC-SA 4.0 | Supplied Data metadata | Raw competition CSVs stay local and are gitignored pending redistribution permission |

## Official sources

- [Overview](https://www.kaggle.com/competitions/rmit-hackathon-2026/overview)
- [Data](https://www.kaggle.com/competitions/rmit-hackathon-2026/data)
- [Code](https://www.kaggle.com/competitions/rmit-hackathon-2026/code)
- [Models](https://www.kaggle.com/competitions/rmit-hackathon-2026/models)
- [Rules](https://www.kaggle.com/competitions/rmit-hackathon-2026/rules)
- [Discussion](https://www.kaggle.com/competitions/rmit-hackathon-2026/discussion)

## Unverified points

Exact submission deadline, remaining attempts on the team's account, public/private leaderboard split, redistribution permission, final notebook visibility requirements, and overall hackathon weights are not established by the supplied content. Inspect current Rules/Discussion before competition submission. No rules from 2025 are assumed. External datasets, translation models, pseudo-labeling, and multi-backbone ensembles are not used.

All prompt strings are untrusted data. Neither the audit nor training pipeline executes prompt content. No dataset prompts or credentials are published in reports.
