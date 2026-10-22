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
    submission_path = ROOT / "artifacts/submission_status.json"
    submission = json.loads(submission_path.read_text(encoding="utf-8")) if submission_path.exists() else {}
    submission_summary = f"Kaggle submissions used: {submission.get('submission_attempts_by_project', 0)}. Status: {submission.get('status', 'not submitted')}. Public score: {submission.get('public_score', 'unknown')}. Private leaderboard score is unknown."
    reports = ROOT / "reports"
    reports.mkdir(exist_ok=True)
    rows = []
    sources = []
    for path in sorted((ROOT / "artifacts").glob("*/metrics.json")):
        metric = json.loads(path.read_text(encoding="utf-8"))
        run = json.loads((path.parent / "run.json").read_text(encoding="utf-8"))
        if "total_seconds" not in run or run.get("execution", {}).get("platform") != "kaggle":
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
        message = """# Kaggle outputs pending

Report: RMIT / SGS / Lục Vị Hoàng Đế. Executed notebook name at submission: UEL_SGS_LucViHoangDe_C1.

Completed Kaggle output files have not been imported into this local report yet. No eligible validation score or leaderboard score is claimed in this report.

Organizer clarifications require all training inside Kaggle, exactly one approved model, no ensembles (including versions of the same model), and no external or synthetic training data. The official entry uses FacebookAI/xlm-roberta-base from the approved Kaggle source. See [verified requirements](../docs/competition.md).

Earlier local experiments were stopped after these clarifications were retrieved. Their checkpoints and [historical reports](local_only/README.md) remain on the machine as ineligible archives. They do not select or initialize the Kaggle entry and must not be submitted.

Next: execute the full private Kaggle notebook, download actual checkpoints and predictions, then regenerate this report and validate the 6,000-row CSV. Competition submissions made by this project: **0**.
"""
        (reports / "RESULTS.md").write_text(message, encoding="utf-8")
        (reports / "results.html").write_text('<!doctype html><html lang="en"><meta charset="utf-8"><title>Kaggle outputs pending</title><style>body{font:17px system-ui;max-width:950px;margin:40px auto;padding:20px}pre{white-space:pre-wrap}</style><pre>' + html.escape(message) + '</pre></html>', encoding="utf-8")
        pd.DataFrame(columns=["run", "model", "macro_f1"]).to_csv(reports / "experiments.csv", index=False)
        for name in ("validation.png", "class_metrics.csv", "length_buckets.csv"):
            (reports / name).unlink(missing_ok=True)
        print("No completed Kaggle runs; local archives excluded.")
        return
    table = pd.DataFrame(rows).sort_values("macro_f1", ascending=False)
    table.to_csv(reports / "experiments.csv", index=False)
    best_name = table.iloc[0]["run"]
    selection_path = ROOT / "artifacts/selection.json"
    if selection_path.exists():
        selected = json.loads(selection_path.read_text(encoding="utf-8"))["selected_run"].replace("\\", "/").rsplit("/", 1)[-1]
        if selected not in set(table["run"]):
            raise ValueError("Selection artifact references an incomplete or non-Kaggle run")
        best_name = selected
    best_path, metric, run = next(item for item in sources if item[0].parent.name == best_name)
    classes = pd.DataFrame([{ "class": label, **metric["class_report"][label]} for label in LABELS])
    lengths = pd.DataFrame([{ "characters": bucket, **stat} for bucket, stat in metric["by_length_characters"].items()])
    classes.to_csv(reports / "class_metrics.csv", index=False)
    lengths.to_csv(reports / "length_buckets.csv", index=False)
    rule_path = ROOT / "docs/rules_evidence.json"
    rules = json.loads(rule_path.read_text(encoding="utf-8")) if rule_path.exists() else {}
    rules_status = ("Rules verified through the official Kaggle API on 2026-10-22. Deadline: 2026-10-22 18:00 Asia/Saigon. " + rules.get("kaggle_code_publication_status", "Check code publication/license status.")) if rules else "Rules verification evidence not available."
    ablation_path = ROOT / "artifacts/kaggle_length_ablation/results.json"
    ablation = json.loads(ablation_path.read_text(encoding="utf-8")) if ablation_path.exists() else None
    ablation_text, ablation_html = "Not completed yet.", "Not completed yet."
    if ablation and ablation.get("execution", {}).get("platform") == "kaggle":
        ablation_table = pd.DataFrame([{k: v for k, v in condition.items() if k != "metrics"} for condition in ablation["conditions"]])
        state = f"{len(ablation_table)}/3 inference conditions completed on {Path(ablation['experiment']).name}. "
        ablation_text = state + ablation["caveat"] + "\n\n```csv\n" + ablation_table.to_csv(index=False, float_format="%.6f", lineterminator="\n") + "```"
        ablation_html = "<p>" + html.escape(state + ablation["caveat"]) + "</p>" + ablation_table.to_html(index=False, border=0, float_format=lambda x: f"{x:.6f}")
    plt.rcParams.update({"font.size": 10, "figure.dpi": 140})
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.7), layout="constrained")
    matrix = np.asarray(metric["confusion_matrix"])
    errors = sorted([(int(matrix[i, j]), LABELS[i], LABELS[j]) for i in range(3) for j in range(3) if i != j and matrix[i, j]], reverse=True)
    error_text = "; ".join(f"{truth} → {predicted}: {count}" for count, truth, predicted in errors)
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
    if final and final.get("execution", {}).get("platform") != "kaggle":
        raise ValueError("Final predictions do not have Kaggle execution provenance")
    final_display = dict(final) if final else None
    if final_display:
        final_display.pop("kaggle_submissions_used_by_this_project", None)
    tokenizer_path = ROOT / "artifacts/tokenizer_verification.json"
    tokenizer_check = json.loads(tokenizer_path.read_text(encoding="utf-8")) if tokenizer_path.exists() else None
    tokenizer_text = (f"Original approved and final-checkpoint tokenizers matched on {tokenizer_check['identical_token_sequences']}/{tokenizer_check['texts_compared']} supplied texts; all backend components matched. This was a local tokenizer-only check, not model inference or training. See [upstream warning issue]({tokenizer_check['primary_source']})." if tokenizer_check and tokenizer_check.get("verified") else "Tokenizer consistency has not yet been verified.")
    final_status = "The final model was retrained on all 14,000 rows. Its final private leaderboard performance is unknown; reported validation scores belong to selection models. The public leaderboard result is recorded separately below." if final else "Final full-data retraining and submission generation have not completed yet."
    table_text = table.to_csv(index=False, float_format="%.6f", lineterminator="\n")
    class_csv = classes.to_csv(index=False, float_format="%.6f", lineterminator="\n")
    length_csv = lengths.to_csv(index=False, float_format="%.6f", lineterminator="\n")
    body = f"""# RMIT_SGS_LucViHoangDe_C1 — Actual experiment results

Report: RMIT / SGS / Lục Vị Hoàng Đế. Executed notebook name at submission: UEL_SGS_LucViHoangDe_C1.

The following numbers come from completed Kaggle training runs. They are validation measurements, not leaderboard scores. Ineligible local experiments are excluded and preserved in reports/local_only.

```csv
{table_text}```

![Validation](validation.png)

Selected configuration run (see artifacts/selection.json for seed confirmation when needed): **{best_name}**, **{metric['macro_f1']:.6f}**.

Per-class precision, recall, F1 and support:

```csv
{class_csv}```

Remaining validation confusions: {error_text}.

Length diagnostics:

```csv
{length_csv}```

Length-bucket F1 uses all three labels even when a small bucket has no examples of one label. It is a diagnostic, not an alternate competition metric.

## Inference length/truncation ablation

{ablation_text}

Validation rows: {run['validation_rows']}. Train rows for selection: {run['train_rows']}. Approximate near duplicates are grouped before splitting. Per-language scores can be noisy for the rare class; small gains need a second seed/split check.

Final submission validation: {json.dumps(final_display, ensure_ascii=False) if final else 'Not generated yet.'}

Tokenizer consistency: {tokenizer_text}

{final_status}

{submission_summary}

{rules_status} See [requirements evidence](../docs/competition.md).
"""
    (reports / "RESULTS.md").write_text(body, encoding="utf-8")
    (reports / "RMIT_SGS_LucViHoangDe_C1.md").write_text(body, encoding="utf-8")
    image = base64.b64encode(chart.read_bytes()).decode()
    html_body = f"""<!doctype html><html lang="vi"><meta charset="utf-8"><title>RMIT SGS — Results</title><style>body{{font:16px system-ui,sans-serif;max-width:1100px;margin:40px auto;padding:0 24px;color:#17212b}}table{{border-collapse:collapse;width:100%;font-size:13px}}th,td{{padding:9px;border-bottom:1px solid #ddd;text-align:left}}img{{max-width:100%}}pre{{white-space:pre-wrap;background:#f3f6f8;padding:18px}}h1{{color:#175c73}}</style><h1>RMIT_SGS_LucViHoangDe_C1</h1><p>Report school: RMIT. Executed notebook: UEL_SGS_LucViHoangDe_C1.</p><p>Completed Kaggle validation runs. Ineligible local archives are excluded. These are not leaderboard scores.</p>{table.to_html(index=False,float_format=lambda x:f'{x:.6f}',border=0)}<img src="data:image/png;base64,{image}" alt="Validation confusion matrix and language metrics"><h2>Per-class metrics</h2>{classes.to_html(index=False,border=0,float_format=lambda x:f'{x:.6f}')}<p>Remaining confusions: {html.escape(error_text)}</p><h2>Length diagnostics</h2>{lengths.to_html(index=False,border=0,float_format=lambda x:f'{x:.6f}')}<p>Bucket F1 includes all three labels; rare/absent labels can make small buckets unstable.</p><h2>Inference truncation ablation</h2>{ablation_html}<h2>Submission checks</h2><pre>{html.escape(json.dumps(final_display,ensure_ascii=False,indent=2)) if final else 'Not generated yet'}</pre><p>{html.escape(final_status)} {html.escape(submission_summary)}</p><h2>Tokenizer verification</h2><p>{html.escape(tokenizer_text)}</p><h2>Rules and publication</h2><p>{html.escape(rules_status)}</p></html>"""
    (reports / "results.html").write_text(html_body, encoding="utf-8")
    (reports / "RMIT_SGS_LucViHoangDe_C1.html").write_text(html_body, encoding="utf-8")
    print(table.to_string(index=False))
    print(reports / "results.html")


if __name__ == "__main__":
    main()
