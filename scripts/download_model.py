"""Download an approved model from the exact Kaggle handle, never a substitute."""
import argparse
import json
import os
from pathlib import Path
import sys

if sys.platform == "win32":
    import truststore
    truststore.inject_into_ssl()

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from protector.constants import APPROVED
from protector.data import sha256, write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", choices=list(APPROVED), default="distilbert/distilbert-base-multilingual-cased")
    parser.add_argument("--cache", default="../cache/kagglehub")
    parser.add_argument("--output", default="artifacts/model_download.json")
    parser.add_argument("--version", type=int)
    args = parser.parse_args()
    os.environ["KAGGLEHUB_CACHE"] = str(Path(args.cache).resolve())
    import kagglehub
    handle = APPROVED[args.model_id] + (f"/{args.version}" if args.version else "")
    path = Path(kagglehub.model_download(handle))
    candidates = list(path.rglob("config.json"))
    if (path / "config.json").exists():
        model_path = path
    elif len(candidates) == 1:
        model_path = candidates[0].parent
    else:
        raise RuntimeError(f"Could not identify a unique model config under {path}")
    files = {str(p.relative_to(model_path)): {"bytes": p.stat().st_size, "sha256": sha256(p)} for p in model_path.rglob("*") if p.is_file() and p.name != "model_provenance.json"}
    result = {"model_id": args.model_id, "kaggle_handle_requested": handle, "resolved_model_path": str(model_path.resolve()), "source_url": "https://www.kaggle.com/models/" + APPROVED[args.model_id], "files": files}
    write_json(model_path / "model_provenance.json", result)
    write_json(args.output, result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
