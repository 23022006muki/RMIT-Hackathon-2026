# RMIT_SGS_LucViHoangDe_C1 — Actual experiment results

Report: RMIT / SGS / Lục Vị Hoàng Đế. Executed notebook name at submission: UEL_SGS_LucViHoangDe_C1.

The following numbers come from completed Kaggle training runs. They are validation measurements, not leaderboard scores. Ineligible local experiments are excluded and preserved in reports/local_only.

```csv
run,model,macro_f1,BENIGN,JAILBREAK,PROMPT_INJECTION,pt,fr,it,en,seconds
kaggle_xlm_roberta_base,FacebookAI/xlm-roberta-base,0.937567,0.955997,0.925059,0.931646,0.911137,0.966972,0.956840,0.957108,576.080000
kaggle_xlm_roberta_weighted,FacebookAI/xlm-roberta-base,0.936549,0.955997,0.925969,0.927681,0.917021,0.962313,0.951642,0.948202,567.720000
kaggle_xlm_roberta_weighted_seed43,FacebookAI/xlm-roberta-base,0.932334,0.952408,0.916021,0.928571,0.904789,0.968517,0.955993,0.933963,568.180000
kaggle_xlm_roberta_base_seed43,FacebookAI/xlm-roberta-base,0.927963,0.954216,0.918216,0.911458,0.896766,0.952338,0.968948,0.933963,567.870000
```

![Validation](validation.png)

Selected configuration run (see artifacts/selection.json for seed confirmation when needed): **kaggle_xlm_roberta_weighted**, **0.936549**.

Per-class precision, recall, F1 and support:

```csv
class,precision,recall,f1-score,support
BENIGN,0.957373,0.954624,0.955997,1741.000000
JAILBREAK,0.920561,0.931442,0.925969,846.000000
PROMPT_INJECTION,0.939394,0.916256,0.927681,203.000000
```

Remaining validation confusions: BENIGN → JAILBREAK: 67; JAILBREAK → BENIGN: 58; PROMPT_INJECTION → BENIGN: 16; BENIGN → PROMPT_INJECTION: 12; PROMPT_INJECTION → JAILBREAK: 1.

Length diagnostics:

```csv
characters,n,macro_f1
0-299,1411,0.956153
300-999,997,0.932600
1000-1999,299,0.922741
2000+,83,0.948232
```

Length-bucket F1 uses all three labels even when a small bucket has no examples of one label. It is a diagnostic, not an alternate competition metric.

## Inference length/truncation ablation

3/3 inference conditions completed on kaggle_xlm_roberta_weighted. Inference-only diagnostics on one Kaggle checkpoint and the same validation split. Final training/inference retain the original configuration; this is not evidence for retraining at another length.

```csv
max_length,truncation,macro_f1,truncated_rows,seconds,BENIGN,JAILBREAK,PROMPT_INJECTION,macro avg,weighted avg
256,head_tail,0.936549,312,11.460000,0.955997,0.925969,0.927681,0.936549,0.944831
256,head,0.935377,312,11.520000,0.955997,0.927059,0.923077,0.935377,0.944827
512,head_tail,0.937474,65,15.180000,0.956797,0.927944,0.927681,0.937474,0.945930
```

Validation rows: 2790. Train rows for selection: 11210. Approximate near duplicates are grouped before splitting. Per-language scores can be noisy for the rare class; small gains need a second seed/split check.

Final submission validation: {"valid": true, "rows": 6000, "columns": ["id", "label"], "label_counts": {"BENIGN": 3774, "JAILBREAK": 1837, "PROMPT_INJECTION": 389}, "sha256": "95e0c3e44e8be2ae5f220e716b1ce58999a44ee3c86fc0e0420dfa87aca56124", "execution": {"platform": "kaggle", "is_kaggle": true, "kernel_run_type": "Batch"}, "checkpoint": "/kaggle/working/artifacts/final/checkpoint", "truncated_test_rows": 619, "bias": [0.0, 0.0, 0.0]}

Tokenizer consistency: Original approved and final-checkpoint tokenizers matched on 20000/20000 supplied texts; all backend components matched. This was a local tokenizer-only check, not model inference or training. See [upstream warning issue](https://github.com/huggingface/transformers/issues/42591).

The final model was retrained on all 14,000 rows. Its final private leaderboard performance is unknown; reported validation scores belong to selection models. The public leaderboard result is recorded separately below.

Kaggle submissions used: 1. Status: complete. Public score: 0.94224. Private leaderboard score is unknown.

Rules verified through the official Kaggle API on 2026-10-22. Deadline: 2026-10-22 18:00 Asia/Saigon. MIT applied; authorized source-link topic published and verified on Kaggle 2026-10-22. See [requirements evidence](../docs/competition.md).
