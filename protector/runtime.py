"""Prevent accidental competition training outside the organizer's Kaggle rule."""
import os
from pathlib import Path
import platform

from .constants import APPROVED
from .data import sha256

ENTRY_MODEL_ID = "FacebookAI/xlm-roberta-base"
COMPETITION_SHA256 = {
    "train.csv": "b6809aea692dbbb4978f45fa603bb001e10ebfbffe9bfbff964e44e2856be795",
    "test.csv": "137925562e9b8403a6f725431489b73b8f267d24423784d1ea0f0a7dc4a455c6",
    "sample_submission.csv": "b25f6188461fb1ad555aa5ce5f7fe7eba356827e8bd541c63b9f59bf6f594f73",
}


def require_kaggle_runtime(model_id=None):
    # KAGGLE_KERNEL_RUN_TYPE is the runtime marker used by Kaggle's own SDK.
    if not (platform.system() == "Linux" and os.environ.get("KAGGLE_KERNEL_RUN_TYPE")
            and Path("/kaggle/input").is_dir() and Path("/kaggle/working").is_dir()):
        raise RuntimeError("Organizer clarification requires ALL competition training/inference in Kaggle. Local/external GPU training and uploading fine-tuned local weights are not allowed. Run UEL_SGS_LucViHoangDe_C1.ipynb on Kaggle.")
    if model_id is not None and model_id != ENTRY_MODEL_ID:
        raise ValueError(f"This entry has selected exactly one backbone: {ENTRY_MODEL_ID}")
    return {"platform": "kaggle", "is_kaggle": True, "kernel_run_type": os.environ["KAGGLE_KERNEL_RUN_TYPE"]}


def verify_competition_input(directory):
    require_kaggle_runtime()
    directory = Path(directory).resolve()
    if not directory.is_relative_to(Path("/kaggle/input")):
        raise ValueError("Use the attached competition input, not an external dataset")
    for name, expected in COMPETITION_SHA256.items():
        if sha256(directory / name) != expected:
            raise ValueError(f"Competition data version/content changed: {name}; review it before training")


def verify_pretrained_input(directory, model_id):
    require_kaggle_runtime(model_id)
    approved = Path("/kaggle/input/models") / APPROVED[ENTRY_MODEL_ID] / "1"
    if not Path(directory).resolve().is_relative_to(approved):
        raise ValueError(f"Use the original approved Kaggle pretrained model at {approved}; local fine-tuned uploads are forbidden")
