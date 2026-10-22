"""Render actual experiment results into local HTML/Markdown/PNG reports."""
import base64
import html
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from protector.constants import LABELS


def main():
    reports = ROOT / "reports"
    reports.mkdir(exist_ok=True)
    rows = []
    sources = []
    for path in sorted((ROOT / "artifacts").glob("*/metrics.json")):
        metric = json.loads(path.read_text(encoding="utf-8"))
        run = json.loads((path.parent / "run.json").read_text(encoding="utf-8"))
        if "total_seconds" not in run:
            continue
        row = {"run": path.parent.name, "model": run["config"]["model_id"], "macro_f1": metric["macro_f1"]}
        for label in LABELS:
            row[label] = metric["class_report"][label]["f1-score"]
        for lang, stat in metric["by_language"].items():
            row[lang] = stat["macro_f1"]
        row["seconds"] = run.get("total_seconds", 0)
        rows.append(row)
        sources.append((path, metric, run))
    if not rows:
        raise RuntimeError("No actual metrics.json exists; train a model first")
    table = pd.DataFrame(rows).sort_values("macro_f1", ascending=False)
    table.to_csv(reports / "experiments.csv", index=False)
    best_name = table.iloc[0]["run"]
    best_path, metric, run = next(item for item in sources if item[0].parent.name == best_name)
    plt.rcParams.update({"font.size": 10, "figure.dpi": 140})
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.7), layout="constrained")
    matrix = np.asarray(metric["confusion_matrix"])
    axes[0].imshow(matrix, cmap="Blues")
    display_labels = ["BENIGN", "JAILBREAK", "PROMPT\nINJECTION"]
    axes[0].set(xticks=range(3), yticks=range(3), xticklabels=display_labels, yticklabels=display_labels, xlabel="Predicted", ylabel="True", title="Validation confusion matrix")
    for i in range(3):
        for j in range(3):
            axes[0].text(j, i, str(matrix[i, j]), ha="center", va="center", color="white" if matrix[i, j] > matrix.max() / 2 else "black")
    langs = list(metric["by_language"])
    values = [metric["by_language"][l]["macro_f1"] for l in langs]
    axes[1].bar(langs, values, color="#1e728f")
    axes[1].set(ylim=(0, 1.05), ylabel="Macro F1 (all three labels)", title="Validation by language")
    for i, value in enumerate(values):
        axes[1].text(i, value + .02, f"{value:.4f}", ha="center")
    fig.suptitle(f"{best_name} | macro F1 = {metric['macro_f1']:.5f}")
    chart = reports / "validation.png"
    fig.savefig(chart)
    plt.close(fig)
    final_path = ROOT / "artifacts/final/submission_validation.json"
    final = json.loads(final_path.read_text(encoding="utf-8")) if final_path.exists() else None
    table_text = table.to_csv(index=False, float_format="%.6f", lineterminator="\n")
    body = f"""# Actual experiment results

Team: UEL / SGS / Lục Vị Hoàng Đế. Notebook: UEL_SGS_LucViHoangDe_C1.

The following numbers come from completed local training runs, not Kaggle leaderboard evaluation.

```csv
{table_text}```

![Validation](validation.png)

Selected by validation macro F1: **{best_name}**, **{metric['macro_f1']:.6f}**.

Validation rows: {run['validation_rows']}. Train rows for selection: {run['train_rows']}. Approximate near duplicates are grouped before splitting. Per-language scores can be noisy for the rare class; small gains need a second seed/split check.

Final submission validation: {json.dumps(final, ensure_ascii=False) if final else 'Not generated yet.'}

Final-model performance is unknown: the full-data model has no held-out labels. Competition submissions made by this project: **0**. Full live Rules remain unverified due Kaggle access/rate limits; see [requirements evidence](../docs/competition.md).
"""
    (reports / "RESULTS.md").write_text(body, encoding="utf-8")
    image = base64.b64encode(chart.read_bytes()).decode()
    html_body = f"""<!doctype html><html lang="vi"><meta charset="utf-8"><title>UEL SGS — Results</title><style>body{{font:16px system-ui,sans-serif;max-width:1100px;margin:40px auto;padding:0 24px;color:#17212b}}table{{border-collapse:collapse;width:100%;font-size:13px}}th,td{{padding:9px;border-bottom:1px solid #ddd;text-align:left}}img{{max-width:100%}}pre{{white-space:pre-wrap;background:#f3f6f8;padding:18px}}h1{{color:#175c73}}</style><h1>UEL_SGS_LucViHoangDe_C1</h1><p>Actual local validation results. These are not leaderboard scores.</p>{table.to_html(index=False,float_format=lambda x:f'{x:.6f}',border=0)}<img src="data:image/png;base64,{image}" alt="Validation confusion matrix and language metrics"><h2>Submission checks</h2><pre>{html.escape(json.dumps(final,ensure_ascii=False,indent=2)) if final else 'Not generated yet'}</pre><p>Macro F1 gives each class equal weight. Final model is retrained on all 14,000 rows; its hidden-test score is unknown. No Kaggle submission has been made. Live Rules could not be verified.</p></html>"""
    (reports / "results.html").write_text(html_body, encoding="utf-8")
    print(table.to_string(index=False))
    print(reports / "results.html")


if __name__ == "__main__":
    main()
