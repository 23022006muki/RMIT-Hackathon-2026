# Two-minute Challenge 2 demo plan

Record a real running app after Tide enrollment and the final Kaggle checkpoint are installed. Keep the exported MP4 below 100 MB. Show the browser and relevant terminal/API output; do not display tokens, passwords, raw competition data or the enrollment URL.

| Time | Screen | Narration |
| --- | --- | --- |
| 0:00–0:18 | Landing page, multilingual aggregate panel | “AI safety teams need to compare patterns across languages, but a shared raw prompt archive creates a new point of control.” |
| 0:18–0:38 | Challenge 1 README/results, one selected model | “Challenge 1 supplies one Kaggle-trained XLM-RoBERTa classifier, evaluated with macro F1 across the three official labels.” State the final verified score only. |
| 0:38–1:05 | Tide login, contributor workspace, type a short demo prompt | “The contributor authenticates with TideCloak. The classifier sees this prompt for inference and returns its label.” |
| 1:05–1:27 | Submit, result, aggregate count rises | “TideCloak self-encrypts the original in the browser. Our server stores ciphertext with the label, so the community can inspect trends.” |
| 1:27–1:48 | Owner case list, decrypt original; optional protected API `401` without token | “The original returns only to its owner, after token verification and Tide decryption. An app administrator cannot simply read the stored text.” |
| 1:48–2:00 | Landing page and GitHub repo | “Signal Commons lets researchers share useful signals without giving one organization the raw archive.” |

If the full Tide flow or model is unavailable, do not simulate success. Show the actual setup state and state the limitation in the submission.
