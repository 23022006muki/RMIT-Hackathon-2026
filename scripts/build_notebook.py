"""Build a self-contained, data-free Kaggle notebook for a selected encoder."""
import argparse
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
from protector.constants import APPROVED, ENCODERS
from protector.data import sha256
from protector.runtime import ENTRY_MODEL_ID


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/xlm_roberta_base.json")
    parser.add_argument("--compare-config", action="append")
    args = parser.parse_args()
    config_paths = list(dict.fromkeys([args.config] + (args.compare_config if args.compare_config is not None else ["configs/xlm_roberta_weighted.json"])))
    configs = [json.loads((ROOT / p).read_text(encoding="utf-8")) for p in config_paths]
    model_id = configs[0]["model_id"]
    if model_id != ENTRY_MODEL_ID or any(c["model_id"] != model_id for c in configs):
        raise ValueError("One approved encoder backbone per notebook")
    team = json.loads((ROOT / "configs/team.json").read_text(encoding="utf-8"))
    title = team["notebook_title"]
    if not re.fullmatch(r"[A-Za-z]+_(SGS|HNI|MEL|MYS)_[A-Za-z0-9_-]+_C1", title):
        raise ValueError("Invalid competition notebook title")
    sources = {p.relative_to(ROOT).as_posix(): p.read_text(encoding="utf-8") for p in sorted((ROOT / "protector").glob("*.py"))}
    for relative in sorted(set(config_paths) | {"configs/team.json"}):
        file = (ROOT / relative).resolve()
        if not file.is_relative_to(ROOT / "configs"):
            raise ValueError("Only repository configuration files can be embedded")
        sources[relative] = file.read_text(encoding="utf-8")
    if not set(config_paths) <= set(sources):
        raise ValueError("Use configuration files inside the repository configs directory")
    payload = base64.b64encode(zlib.compress(json.dumps(sources).encode("utf-8"), 9)).decode("ascii")
    cells = [nbf.v4.new_markdown_cell(f"# {title}\n\n**Chosen approved backbone:** `{model_id}`\n\n**Team:** {team['display_name']} — {team['university']} / {team['campus']}. Primary configuration: `{args.config}`.\n\nFine-tune one approved encoder, evaluate macro F1 on a group-aware split, then retrain on all training rows. Attach the competition and the [approved Kaggle model](https://www.kaggle.com/models/{APPROVED[model_id]}) in Input and enable a GPU. Source code is embedded; competition IDs, labels, prompts, predictions and split data are not embedded. No GitHub checkout or external pretrained backbone is fetched. This notebook creates a CSV and never submits to Kaggle.\n\nPrepare compatible dependencies before an offline run: PyTorch >=2.8,<3, Transformers >=4.56,<5, NumPy >=2.1,<3, pandas >=2.2,<3, scikit-learn >=1.6,<2, safetensors >=0.5. If packages are missing, install compatible versions from PyPI while internet is enabled and restart the kernel. Keep a compatible CUDA PyTorch build.\n\nCompetition Overview, Data, Challenge and Foundational Rules were verified through the official Kaggle API on 2026-10-22. Deadline: 18:00 Asia/Saigon on 2026-10-22. Public code must also be shared on the competition Kaggle forum/notebooks under an OSI-approved license; check repository publication/license status before submission. Organizer replies require all training inside Kaggle, exactly one approved model, no combining predictions from different checkpoints, and no external or synthetic training data. Only original approved pretrained weights initialize this notebook. Earlier local experiments are ineligible archives and are never loaded here. See Discussion topics 751573, 751570 and 751566. No external data, translation model or pseudo-labeling is used.")]
    cells.append(nbf.v4.new_code_cell(f"""import os, sys, json, base64, zlib
from pathlib import Path
from packaging.version import Version
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
MODEL_ID = {model_id!r}
CONFIG_PATHS = {config_paths!r}
MODEL_DIR = None  # Set explicitly if automatic discovery is ambiguous.
DATA_DIR = None
assert os.environ.get('KAGGLE_KERNEL_RUN_TYPE') and Path('/kaggle/working').is_dir(), 'Organizer rule: all competition training must execute inside Kaggle.'
WORK = Path('/kaggle/working')
os.chdir(WORK)
import torch, transformers, numpy, pandas, sklearn
for name, current, lo, hi in [('torch',torch.__version__,'2.8','3'), ('transformers',transformers.__version__,'4.56','5'), ('numpy',numpy.__version__,'2.1','3'), ('pandas',pandas.__version__,'2.2','3'), ('sklearn',sklearn.__version__,'1.6','2')]:
    assert Version(lo) <= Version(current) < Version(hi), f'Prepare compatible {{name}}; found {{current}}'
assert torch.cuda.is_available(), 'Enable GPU for a full training run.'
print({{'torch':torch.__version__, 'transformers':transformers.__version__, 'GPU':torch.cuda.get_device_name(0)}})"""))
    cells.append(nbf.v4.new_code_cell(f"""# Source matches protector/ and configs/ in the repository.
payload = {payload!r}
embedded = json.loads(zlib.decompress(base64.b64decode(payload)))
assert all(p.startswith(('protector/', 'configs/')) for p in embedded)
for relative, contents in embedded.items():
    path = (WORK / relative).resolve()
    assert path.is_relative_to(WORK.resolve())
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(contents, encoding='utf-8')
sys.path.insert(0, str(WORK))
print('Embedded source files:', sorted(embedded))"""))
    cells.append(nbf.v4.new_code_cell("""from protector.constants import APPROVED, ENCODERS
from protector.data import read_data, sha256, audit_data, validate_submission
from protector.training import train_experiment, finalize
from protector.runtime import require_kaggle_runtime, verify_competition_input, verify_pretrained_input
require_kaggle_runtime(MODEL_ID)
assert MODEL_ID in ENCODERS
assert all(json.loads((WORK/p).read_text())['model_id'] == MODEL_ID for p in CONFIG_PATHS)
input_root = Path('/kaggle/input')
if DATA_DIR is None:
    roots = [p.parent for p in input_root.rglob('train.csv') if (p.parent/'test.csv').exists() and (p.parent/'sample_submission.csv').exists()]
    assert len(roots) == 1, f'Set DATA_DIR explicitly; candidates: {roots}'
    DATA_DIR = roots[0]
DATA_DIR = Path(DATA_DIR)
verify_competition_input(DATA_DIR)
train, test, sample = read_data(DATA_DIR)
audit_data(DATA_DIR, WORK/'artifacts/audit')
if MODEL_DIR is None:
    slug = APPROVED[MODEL_ID].split('/')[1]
    candidates = [p.parent for p in input_root.rglob('config.json') if slug in str(p.parent).lower()]
    assert len(candidates) == 1, f'Attach the approved Kaggle model or set MODEL_DIR. Candidates: {candidates}'
    MODEL_DIR = candidates[0]
MODEL_DIR = Path(MODEL_DIR)
verify_pretrained_input(MODEL_DIR, MODEL_ID)
assert (MODEL_DIR/'config.json').exists()
print('Data:', DATA_DIR, 'Model:', MODEL_DIR, 'Rows:', len(train), len(test))"""))
    cells.append(nbf.v4.new_code_cell("""runs = []
for config_path in CONFIG_PATHS:
    output = WORK/'artifacts'/('kaggle_' + Path(config_path).stem)
    if (output/'history.json').exists():
        meta = json.loads((output/'run.json').read_text())
        assert meta.get('execution', {}).get('platform') == 'kaggle'
        assert 'total_seconds' in meta, 'Incomplete earlier run: use a fresh output directory.'
        assert meta['provenance']['train_file_sha256'] == sha256(DATA_DIR/'train.csv')
        assert meta['provenance']['split_sha256'] == sha256(WORK/'artifacts/audit/split.csv')
        assert meta['provenance']['config_file_sha256'] == sha256(WORK/config_path)
    else:
        train_experiment(DATA_DIR, WORK/'artifacts/audit/split.csv', WORK/config_path, MODEL_DIR, output)
    metric = json.loads((output/'metrics.json').read_text())
    runs.append((metric['macro_f1'], output))
    print(output.name, metric)
selected_score, selected_run = max(runs, key=lambda item: item[0])
if len(runs) == 2 and abs(runs[0][0] - runs[1][0]) < 0.005:
    # Compare optimization stability; predictions/checkpoints are never combined.
    confirmed = []
    for config_path, (score42, original_run) in zip(CONFIG_PATHS, runs):
        config = json.loads((WORK/config_path).read_text())
        config['seed'] = 43
        confirmation_config = WORK/'configs'/(Path(config_path).stem + '_seed43.json')
        confirmation_config.write_text(json.dumps(config, indent=2))
        confirmation_run = WORK/'artifacts'/('kaggle_' + confirmation_config.stem)
        if not (confirmation_run/'run.json').exists():
            train_experiment(DATA_DIR, WORK/'artifacts/audit/split.csv', confirmation_config, MODEL_DIR, confirmation_run)
        confirmation_meta = json.loads((confirmation_run/'run.json').read_text())
        assert 'total_seconds' in confirmation_meta and confirmation_meta['execution']['platform'] == 'kaggle'
        score43 = json.loads((confirmation_run/'metrics.json').read_text())['macro_f1']
        confirmed.append(((score42 + score43)/2, original_run))
    _, selected_run = max(confirmed, key=lambda item: item[0])
    selected_score = json.loads((selected_run/'metrics.json').read_text())['macro_f1']
    (WORK/'artifacts/selection.json').write_text(json.dumps({'method':'mean validation F1 across seeds 42 and 43; no ensembling', 'scores':[(s,str(p)) for s,p in confirmed], 'selected_run':str(selected_run)}, indent=2))
else:
    (WORK/'artifacts/selection.json').write_text(json.dumps({'method':'validation F1; difference >= 0.005 or one configuration', 'selected_run':str(selected_run)}, indent=2))
print('Selected single configuration:', selected_run, 'seed-42 validation macro F1:', selected_score)"""))
    cells.append(nbf.v4.new_code_cell("""from protector.diagnostics import evaluate_lengths
evaluate_lengths(selected_run, DATA_DIR, WORK/'artifacts/audit/split.csv', WORK/'artifacts/kaggle_length_ablation')"""))
    cells.append(nbf.v4.new_code_cell("""final_dir = WORK/'artifacts/final'
if not (final_dir/'submission_validation.json').exists():
    finalize(DATA_DIR, selected_run, MODEL_DIR, final_dir)
else:
    final_meta = json.loads((final_dir/'run.json').read_text())
    assert final_meta.get('execution', {}).get('platform') == 'kaggle'
    assert final_meta['provenance']['selected_experiment'] == str(selected_run.resolve()), 'Existing final output used another experiment; use a fresh output directory.'
validation = validate_submission(final_dir/'submission.csv', DATA_DIR)
print(json.dumps(validation, indent=2))
from IPython.display import display, FileLink
display(FileLink(str(final_dir/'submission.csv')))
print('No Kaggle submission was made. Review the CSV before authorizing submission.')"""))
    cells.append(nbf.v4.new_markdown_cell("Validation scores belong to the selection model trained on the training partition. The final model is retrained on all 14,000 rows for the selected epoch count; its hidden-test score is unknown. CSV validity does not establish hidden-test performance. Class order: BENIGN, JAILBREAK, PROMPT_INJECTION. Prompt strings are always untrusted data. CUDA attention can cause small numerical differences even with fixed seeds."))
    for cell in cells:
        cell.id = hashlib.sha256(cell.source.encode("utf-8")).hexdigest()[:12]
        if cell.cell_type == "code":
            compile(cell.source, "<notebook cell>", "exec")
    notebook = nbf.v4.new_notebook(cells=cells, metadata={"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python"}, "title": title})
    nbf.validate(notebook)
    output = ROOT / "notebooks" / f"{title}.ipynb"
    output.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(notebook, output)
    print(output)
    print("sha256", sha256(output))


if __name__ == "__main__":
    main()
