import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

from .constants import LABEL2ID
from .data import write_json


def calibrate(experiment):
    """Tune on half the validation data and assess on the other half.

    This optional analysis must not be confused with a new independent test:
    checkpoint selection already used the whole validation set.
    """
    experiment = Path(experiment)
    frame = pd.read_csv(experiment / "validation_predictions.csv")
    logits = frame[[f"logit_{label}" for label in LABEL2ID]].to_numpy()
    y = frame.label.map(LABEL2ID).to_numpy()
    tune, check = train_test_split(np.arange(len(frame)), test_size=0.5, random_state=137,
                                  stratify=frame.language + "_" + frame.label)
    def metric(idx, bias):
        return float(f1_score(y[idx], (logits[idx] + bias).argmax(1), labels=[0, 1, 2], average="macro", zero_division=0))
    zero = np.zeros(3)
    best_bias, best_f1 = zero, metric(tune, zero)
    for j in (-0.5, 0.0, 0.5):
        for p in (-1.0, -0.5, 0.0, 0.5, 1.0):
            bias = np.array([0.0, j, p])
            value = metric(tune, bias)
            if value > best_f1 + 1e-8:
                best_bias, best_f1 = bias, value
    base_check, biased_check = metric(check, zero), metric(check, best_bias)
    selected = best_bias if biased_check >= base_check + 0.002 else zero
    result = {"selected_bias": selected.tolist(), "candidate_bias": best_bias.tolist(),
              "tuning_macro_f1_baseline": metric(tune, zero), "tuning_macro_f1_candidate": best_f1,
              "assessment_macro_f1_baseline": base_check, "assessment_macro_f1_candidate": biased_check,
              "accepted": bool(np.any(selected)), "tuning_rows": len(tune), "assessment_rows": len(check),
              "caveat": "Assessment half was held out from bias tuning, but was previously used in checkpoint selection. Confirm improvements on a second split/seed before relying on small changes. No per-language threshold fitting or test labels used."}
    write_json(experiment / "calibration.json", result)
    print(json.dumps(result, indent=2), flush=True)
    return result
