# Competition requirements and evidence

Team: **Lục Vị Hoàng Đế**, university **UEL**, campus **SGS**. Notebook: **UEL_SGS_LucViHoangDe_C1**.

On **2026-10-22**, the official Kaggle SDK successfully read `CompetitionApiService.GetCompetition` and `ListCompetitionPages`. This supersedes the earlier browser rate-limit limitation. Source page hashes and retrieval metadata are in [rules_evidence.json](rules_evidence.json); full source responses are kept locally under `artifacts/sources/` and are not republished.

| Requirement | Official evidence | Implementation/status |
|---|---|---|
| Three exact labels; pt/fr/it/en | Data, Challenge, Evaluation pages; local audit | Fixed mapping and schema checks |
| 14,000 train and 6,000 test rows | Data page and CSV files | Strict size, schema, blank-field and ID checks |
| Macro F1, equal weight for each class | Evaluation page and API metric | Explicit three-class macro F1; precision/recall/F1 by class |
| One approved pretrained backbone for the final solution | Challenge page, approved model list | Allowlist; exact approved Kaggle DistilBERT and XLM-R source versions downloaded for separate experiments; no ensemble |
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
| Public code must also be shared on the competition's Kaggle forum or notebooks, with an OSI-approved license permitting commercial use | Foundational Rules 6(b), 6(c) | GitHub repo verified public. Kaggle code-sharing post and MIT proposal await user authorization; see [draft](kaggle_discussion_draft.md). No full-compliance claim while this action is pending |
| Hidden private leaderboard determines final Kaggle ranking | Foundational Rules 7 | Choose using validation; public leaderboard is not a training target. Exact public/private proportions remain unknown |
| Dataset license | API and Data metadata | CC BY-NC-SA 4.0; applies to data, not automatically to the team's source code |

Foundational Rules state that they control a conflict with competition-specific terms. No 2025 rule or weight is assumed. The current pages do not establish total RMIT hackathon challenge weights or the account's remaining submission count.

## Public sharing and local artifacts

The user authorized GitHub publication. An automatic approval review separately required specific permission for exporting ID-level competition artifacts. Source and aggregate metrics are published; raw CSVs, splits, near-duplicate ID pairs, validation predictions, final prediction CSVs and checkpoints remain local pending the relevant authorization. Do not interpret this local file policy as proof that the competition allows public redistribution of its data.

The draft Kaggle post only shares the already-public GitHub source link. Posting it does not submit a prediction CSV. No Kaggle post, notebook publication, rule acceptance or competition submission has been performed by this project. A code-license proposal has been prepared without applying it until the user decides.

## Discussion clarifications still unresolved

The official FAQ topic was read through the SDK. It asks participants to share notebooks with specified people or the Kaggle Group feature; reviewer identities and the exact scoring workflow are not supplied in that text. The following current discussion questions were also read, but their answers could not be retrieved: the public comment API returned empty arrays and the browser remained rate-limited. Do not treat the questions as organizer permission:

- [Local/external GPU training and notebook reruns](https://www.kaggle.com/competitions/rmit-hackathon-2026/discussion/751573)
- [External or synthetic training data](https://www.kaggle.com/competitions/rmit-hackathon-2026/discussion/751570)
- [Combining models and notebook verification](https://www.kaggle.com/competitions/rmit-hackathon-2026/discussion/751566)

Local experiment scores are valid local measurements, but do not alone establish competition acceptance of a locally trained checkpoint. The generated notebook trains the complete selected pipeline from the approved Kaggle pretrained model and supplied data; it does not require importing local fine-tuned weights. Confirm organizer guidance or execute the full notebook on Kaggle before relying on a local-only CSV for scoring.

## Official sources

- [Overview and Challenge](https://www.kaggle.com/competitions/rmit-hackathon-2026/overview)
- [Data](https://www.kaggle.com/competitions/rmit-hackathon-2026/data)
- [Code](https://www.kaggle.com/competitions/rmit-hackathon-2026/code)
- [Models](https://www.kaggle.com/competitions/rmit-hackathon-2026/models)
- [Rules and Foundational Rules](https://www.kaggle.com/competitions/rmit-hackathon-2026/rules)
- [Official Discussion](https://www.kaggle.com/competitions/rmit-hackathon-2026/discussion/748423)

Prompt strings are untrusted data and are never executed. External datasets, translation models, pseudo-labeling, multi-backbone ensembles and human-labeled test records are not used.
