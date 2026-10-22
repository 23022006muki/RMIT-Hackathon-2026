# UEL_SGS_LucViHoangDe_C1 — XLM-RoBERTa training code

Our team (UEL / SGS / Lục Vị Hoàng Đế) has made its Challenge 1 source code available here:

https://github.com/23022006muki/RMIT-Hackathon-2026

The official entry notebook selects FacebookAI/xlm-roberta-base from the approved Kaggle model source aisuko/xlm-roberta-base/transformers/default/1. It runs training and inference inside Kaggle using only the supplied competition data. It includes group-aware validation stratified by language and label, cross-entropy/class-weight comparison, selection of one configuration, full-data retraining, and strict submission CSV validation. It never combines predictions from multiple checkpoints.

The first-party source code is licensed under MIT. Competition data and pretrained model weights retain their original licenses; see the repository license scope.

This post shares the public source link with competition participants for Foundational Rules section 6(b). It contains no raw prompts, ID-level predictions, checkpoint files, or private notebook execution links. Posting this link does not submit a prediction CSV.
