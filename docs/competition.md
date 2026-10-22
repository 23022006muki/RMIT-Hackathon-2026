# Competition requirements and evidence

Team: **Lục Vị Hoàng Đế**, university **UEL**, campus **SGS**. Notebook: **UEL_SGS_LucViHoangDe_C1**.

On **2026-10-22**, the official Kaggle SDK successfully read `CompetitionApiService.GetCompetition` and `ListCompetitionPages`. This supersedes the earlier browser rate-limit limitation. Source page hashes and retrieval metadata are in [rules_evidence.json](rules_evidence.json); full source responses are kept locally under `artifacts/sources/` and are not republished.

| Requirement | Official evidence | Implementation/status |
|---|---|---|
| Three exact labels; pt/fr/it/en | Data, Challenge, Evaluation pages; local audit | Fixed mapping and schema checks |
| 14,000 train and 6,000 test rows | Data page and CSV files | Strict size, schema, blank-field and ID checks |
| Macro F1, equal weight for each class | Evaluation page and API metric | Explicit three-class macro F1; precision/recall/F1 by class |
| One approved pretrained backbone for the final solution | Challenge page, approved model list | Official entry selects FacebookAI/xlm-roberta-base only, from approved Kaggle version 1; no ensembles, including same-backbone checkpoints. Earlier local experiments are ineligible archives. |
| Fine-tune on supplied training data | Challenge page | Held-out selection, then full-data retraining; no external dataset |
| Model name at notebook top | Challenge page | Generated first markdown cell |
| Naming `<university>_<campus>_<team>_C1` | Overview abstract | UEL_SGS_LucViHoangDe_C1; unaccented team identifier |
| 6,000 rows, exact id,label columns and allowed labels | Evaluation page | Strict validator; sample-submission order retained |
| Submission deadline | GetCompetition API | **2026-10-22 18:00 Asia/Saigon** = 11:00 UTC; check for changes before submission |
| Submission allowance | API: six daily; Challenge text: six attempts | Conservative project budget of six; project has made zero submissions; actual account allowance not inspected |
| Maximum team size | GetCompetition API | Four; participant must confirm their real Kaggle team membership |
| First valid submission earns full Challenge 1 marks | Challenge page | Does not establish a high leaderboard score or overall hackathon weight |
| Submit notebook for higher score | Competition-specific Rules page | Exact custom instruction: “Submit notebooks to get higher score.” Named notebook is prepared; publication/submission requires participant action or authorization |
| No manually labeling validation/test records for submissions | Foundational Rules 4(b) | Original train labels retained; no manual prediction overrides or test relabeling |
| One account per person and one team | Foundational Rules 5 | No proxy account or attempt-limit workarounds |
| No private code/data sharing outside the team | Foundational Rules 5(d), 6(a) | Do not privately transfer competition code/data to another team |
| Public code must also be shared on the competition's Kaggle forum or notebooks, with an OSI-approved license permitting commercial use | Foundational Rules 6(b), 6(c) | GitHub repo verified public. User authorized MIT and the Kaggle source-link post. MIT is applied and [public source-link topic](https://www.kaggle.com/competitions/rmit-hackathon-2026/discussion/751585) was published and verified. |
| Hidden private leaderboard determines final Kaggle ranking | Foundational Rules 7 | Choose using validation; public leaderboard is not a training target. Exact public/private proportions remain unknown |
| Dataset license | API and Data metadata | CC BY-NC-SA 4.0; applies to data, not automatically to the team's source code |

Foundational Rules state that they control a conflict with competition-specific terms. No 2025 rule or weight is assumed. The current pages do not establish total RMIT hackathon challenge weights or the account's remaining submission count.

## Public sharing and local artifacts

The user authorized GitHub publication. An automatic approval review separately required specific permission for exporting ID-level competition artifacts. Source and aggregate metrics are published; raw CSVs, splits, near-duplicate ID pairs, validation predictions, final prediction CSVs and checkpoints remain local pending the relevant authorization. Do not interpret this local file policy as proof that the competition allows public redistribution of its data.

The published Kaggle post shares the already-public GitHub source link and describes the XLM-RoBERTa pipeline. Posting it did not submit a prediction CSV. The user explicitly authorized MIT for first-party code and the Kaggle source-link post. MIT has been applied; the [public source-link topic](https://www.kaggle.com/competitions/rmit-hackathon-2026/discussion/751585) was published and verified. No public notebook release, rule acceptance or competition CSV submission has been performed by this project.

## Verified organizer clarifications

Official `CompetitionApiService.ListTopicMessages` returned the organizer replies; this supersedes the previous empty generic-comment API responses.

| Clarification | Official reply | Enforcement |
|---|---|---|
| All training must occur inside Kaggle; local/external GPU training and uploading local fine-tuned weights are forbidden | [Topic 751573, reply 3542538](https://www.kaggle.com/competitions/rmit-hackathon-2026/discussion/751573) | Runtime guards, original approved Kaggle input, and checkpoint-origin metadata |
| Use only supplied training data; no external data or synthetic data generation | [Topic 751570, reply 3542539](https://www.kaggle.com/competitions/rmit-hackathon-2026/discussion/751570) | Exact competition input hashes; no augmentation or external data |
| Exactly one approved model; no ensembles, including multiple fine-tuned versions of one model | [Topic 751566, reply 3542536](https://www.kaggle.com/competitions/rmit-hackathon-2026/discussion/751566) | Only XLM-RoBERTa configurations appear in the official notebook; select one loss/configuration; never combine predictions |
| Complete training/inference notebook required; organizers can access notebooks; high-ranking notebooks may be public | Same replies and official FAQ | Self-contained notebook, exact team title, complete training from approved original pretrained input |

Earlier local DistilBERT and partial XLM-R experiments were run before these answers were successfully retrieved. They were stopped and explicitly marked ineligible. Their model files remain locally for transparency. Reports under `reports/local_only` are historical measurements, not eligible entry scores. No checkpoint or CSV from those runs may be submitted or uploaded as Kaggle model initialization. The official entry is trained afresh in Kaggle.

The official FAQ asks participants to share notebooks with specific people or via Kaggle Group; reviewer identities are not supplied in that text. Confirm the participant's actual organizer/team sharing configuration before claiming verification access is complete. Local random-model unit fixtures contain no competition data and only test software mechanics; they are not training-data generation for the entry.

## Official sources

- [Overview and Challenge](https://www.kaggle.com/competitions/rmit-hackathon-2026/overview)
- [Data](https://www.kaggle.com/competitions/rmit-hackathon-2026/data)
- [Code](https://www.kaggle.com/competitions/rmit-hackathon-2026/code)
- [Models](https://www.kaggle.com/competitions/rmit-hackathon-2026/models)
- [Rules and Foundational Rules](https://www.kaggle.com/competitions/rmit-hackathon-2026/rules)
- [Official Discussion](https://www.kaggle.com/competitions/rmit-hackathon-2026/discussion/748423)

Prompt strings are untrusted data and are never executed. External datasets, translation models, pseudo-labeling, multi-backbone ensembles and human-labeled test records are not used.
