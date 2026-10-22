# Actual experiment results

Team: UEL / SGS / Lục Vị Hoàng Đế. Notebook: UEL_SGS_LucViHoangDe_C1.

The following numbers come from completed local training runs, not Kaggle leaderboard evaluation.

```csv
run,model,macro_f1,BENIGN,JAILBREAK,PROMPT_INJECTION,pt,fr,it,en,seconds
distilbert_ce,distilbert/distilbert-base-multilingual-cased,0.908549,0.940307,0.895260,0.890080,0.877467,0.953065,0.945971,0.849625,555.450000
distilbert_weighted,distilbert/distilbert-base-multilingual-cased,0.906328,0.938462,0.892794,0.887728,0.876938,0.946834,0.931041,0.884757,541.670000
```

![Validation](validation.png)

Current best among completed runs by validation macro F1: **distilbert_ce**, **0.908549**.

Per-class precision, recall, F1 and support:

```csv
class,precision,recall,f1-score,support
BENIGN,0.928171,0.952765,0.940307,1736.000000
JAILBREAK,0.909631,0.881336,0.895260,868.000000
PROMPT_INJECTION,0.937853,0.846939,0.890080,196.000000
```

Remaining validation confusions: JAILBREAK → BENIGN: 103; BENIGN → JAILBREAK: 71; PROMPT_INJECTION → BENIGN: 25; BENIGN → PROMPT_INJECTION: 11; PROMPT_INJECTION → JAILBREAK: 5.

Length diagnostics:

```csv
characters,n,macro_f1
0-299,1369,0.945027
300-999,1037,0.882523
1000-1999,317,0.877666
2000+,77,0.915670
```

Length-bucket F1 uses all three labels even when a small bucket has no examples of one label. It is a diagnostic, not an alternate competition metric.

## Inference length/truncation ablation

Not completed yet.

Validation rows: 2800. Train rows for selection: 11200. Approximate near duplicates are grouped before splitting. Per-language scores can be noisy for the rare class; small gains need a second seed/split check.

Final submission validation: Not generated yet.

Final full-data retraining and submission generation have not completed yet. Competition submissions made by this project: **0**.

Rules verified through the official Kaggle API on 2026-10-22. Deadline: 2026-10-22 18:00 Asia/Saigon. Pending user authorization; GitHub repository verified public. No Kaggle post or notebook publication made by this project. See [requirements evidence](../docs/competition.md).
