"""Build a self-contained Kaggle notebook: no GitHub checkout needed at runtime."""
import base64
import hashlib
import json
from pathlib import Path
import re
import sys
import zlib

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from protector.data import sha256


def main():
    team = json.loads((ROOT / "configs/team.json").read_text(encoding="utf-8"))
    title = team["notebook_title"]
    if not re.fullmatch(r"[A-Za-z]+_(SGS|HNI|MEL|MYS)_[A-Za-z0-9_-]+_C1", title):
        raise ValueError("Invalid competition notebook title")
    sources = {str(p.relative_to(ROOT)).replace("\\", "/"): p.read_text(encoding="utf-8") for p in sorted((ROOT / "protector").glob("*.py"))}
    for file in ("distilbert_ce.json", "distilbert_weighted.json", "xlm_roberta_base.json", "team.json"):
        sources["configs/" + file] = (ROOT / "configs" / file).read_text(encoding="utf-8")
    payload = base64.b64encode(zlib.compress(json.dumps(sources).encode("utf-8"), 9)).decode("ascii")
    cells = []
    cells.append(nbf.v4.new_markdown_cell(f"# {title}\n\n**Chosen approved backbone:** `distilbert/distilbert-base-multilingual-cased`\n\n**Team:** {team['display_name']} — UEL / SGS.\n\nFine-tune one approved multilingual encoder; compare cross-entropy against square-root class weighting on a fixed group-aware validation split, then retrain the selected configuration on all training rows. Metric: macro F1 over the three exact competition labels.\n\nAttach the competition and the approved [Kaggle model](https://www.kaggle.com/models/aisuko/distilbert-base-multilingual-cased/transformers/default) in Input, enable a GPU, then Run All. The notebook includes its Python source and computes split IDs locally and does not fetch GitHub or an external pretrained backbone. Dependencies should be prepared before an offline run. It creates a CSV but never submits to the leaderboard.\n\nThe source Overview/Data were supplied by the participant. Live Rules could not be independently verified because Kaggle returned rate limits; check Rules/Discussion before entering. No external data, translation model, pseudo-labeling, or multi-backbone ensemble is used."))
    cells.append(nbf.v4.new_code_cell("""import os, sys, json, base64, zlib
from pathlib import Path
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
MODEL_ID = 'distilbert/distilbert-base-multilingual-cased'
# Set an explicit attached-model directory only if discovery finds multiple inputs.
MODEL_DIR = None
DATA_DIR = None
RUN_WEIGHTED_COMPARISON = True
WORK = Path('/kaggle/working') if Path('/kaggle/working').exists() else Path.cwd()
os.chdir(WORK)
import torch, transformers, numpy, pandas, sklearn
assert torch.cuda.is_available(), 'Enable GPU before a full training run.'
assert transformers.__version__.split('.')[0] == '4', 'Use transformers>=4.56,<5 as in requirements.txt.'
print({'torch': torch.__version__, 'transformers': transformers.__version__, 'GPU': torch.cuda.get_device_name(0)})"""))
    cells.append(nbf.v4.new_code_cell(f"""# Embedded, reviewable source; no downloaded executable code.
payload = {payload!r}
embedded = json.loads(zlib.decompress(base64.b64decode(payload)))
for relative, contents in embedded.items():
    path = WORK / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(contents, encoding='utf-8')
sys.path.insert(0, str(WORK))
print('Embedded files:', sorted(embedded))"""))
    cells.append(nbf.v4.new_code_cell(f"""from protector.constants import APPROVED
from protector.data import read_data, sha256, write_json, audit_data
from protector.training import train_experiment, finalize
assert MODEL_ID == 'distilbert/distilbert-base-multilingual-cased', 'This notebook is configured for the explicitly selected DistilBERT backbone.'
input_root = Path('/kaggle/input')
if DATA_DIR is None:
    roots = [p.parent for p in input_root.rglob('train.csv') if (p.parent/'test.csv').exists() and (p.parent/'sample_submission.csv').exists()]
    assert len(roots) == 1, f'Set DATA_DIR explicitly; candidates: {{roots}}'
    DATA_DIR = roots[0]
DATA_DIR = Path(DATA_DIR)
train, test, sample = read_data(DATA_DIR)
audit_data(DATA_DIR, WORK/'artifacts/audit')
if MODEL_DIR is None:
    slug = APPROVED[MODEL_ID].split('/')[1]
    candidates = [p.parent for p in input_root.rglob('config.json') if slug in str(p.parent).lower()]
    assert len(candidates) == 1, f'Attach the approved Kaggle model or set MODEL_DIR explicitly. Candidates: {{candidates}}'
    MODEL_DIR = candidates[0]
MODEL_DIR = Path(MODEL_DIR)
assert (MODEL_DIR/'config.json').exists()
print('Data:', DATA_DIR, 'Model:', MODEL_DIR, 'Rows:', len(train), len(test))"""))
    cells.append(nbf.v4.new_code_cell("""runs = []
configs = ['distilbert_ce'] + (['distilbert_weighted'] if RUN_WEIGHTED_COMPARISON else [])
for name in configs:
    output = WORK/'artifacts'/name
    if not (output/'metrics.json').exists():
        train_experiment(DATA_DIR, WORK/'artifacts/audit/split.csv', WORK/f'configs/{name}.json', MODEL_DIR, output)
    metric = json.loads((output/'metrics.json').read_text())
    runs.append((metric['macro_f1'], output))
    print(name, metric)
selected_score, selected_run = max(runs, key=lambda item: item[0])
print('Selected:', selected_run, 'validation macro F1:', selected_score)
print('Small differences require confirmation on another seed/split before making strong claims.')"""))
    cells.append(nbf.v4.new_code_cell("""final_dir = WORK/'artifacts/final'
if not (final_dir/'submission_validation.json').exists():
    finalize(DATA_DIR, selected_run, MODEL_DIR, final_dir)
validation = json.loads((final_dir/'submission_validation.json').read_text())
print(json.dumps(validation, indent=2))
from IPython.display import display, FileLink
display(FileLink(str(final_dir/'submission.csv')))
print('No Kaggle submission was made. Review and submit manually when authorized.')"""))
    cells.append(nbf.v4.new_markdown_cell("## Output interpretation\n\nValidation metrics belong to selection models trained on the training partition. The final checkpoint is trained again on all 14,000 rows for the selected number of epochs; its generalization score is unknown until Kaggle evaluates it. A valid CSV is not proof of a high hidden-test score. Class order is fixed as BENIGN, JAILBREAK, PROMPT_INJECTION. Original prompts are treated solely as untrusted training data."))
    notebook = nbf.v4.new_notebook(cells=cells, metadata={"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python", "version": "3.11"}, "title": title})
    nbf.validate(notebook)
    for cell in cells:
        if cell.cell_type == "code":
            compile(cell.source, "<notebook cell>", "exec")
    output = ROOT / "notebooks" / f"{title}.ipynb"
    output.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(notebook, output)
    print(output)
    print('sha256', sha256(output))


if __name__ == "__main__":
    main()
