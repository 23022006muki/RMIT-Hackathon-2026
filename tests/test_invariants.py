import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from protector.constants import LABELS
from protector.data import duplicate_groups, make_split, read_data, validate_submission
from protector.metrics import score


class InvariantTests(unittest.TestCase):
    def test_macro_f1_counts_absent_prediction_classes(self):
        frame = pd.DataFrame({"label": LABELS, "language": ["pt"] * 3, "text": ["a", "b", "c"]})
        metric, _ = score(frame, np.array([[5, 0, 0]] * 3))
        self.assertAlmostEqual(metric["macro_f1"], 1 / 6)
        self.assertEqual(np.asarray(metric["confusion_matrix"]).sum(), 3)

    def test_normalized_duplicate_stays_in_same_partition(self):
        rows = []
        for language in ("pt", "fr", "it", "en"):
            for label in LABELS:
                for i in range(12):
                    rows.append({"id": f"{language}_{label}_{i}", "language": language, "label": label, "text": f"{language} {label} unique {i}"})
        frame = pd.DataFrame(rows)
        frame.loc[1, "text"] = "  " + frame.loc[0, "text"].upper() + "  "
        groups, _ = duplicate_groups(frame)
        split = make_split(frame, groups)
        self.assertEqual(groups[0], groups[1])
        self.assertEqual(split.loc[0, "partition"], split.loc[1, "partition"])
        self.assertFalse(set(split[split.partition == "train"].group) & set(split[split.partition == "validation"].group))

    def test_submission_rejects_duplicates_and_extra_columns(self):
        temp_root = Path(__file__).resolve().parents[1] / ".test_tmp"
        temp_root.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=temp_root) as td:
            path = Path(td)
            tr = pd.DataFrame({"id": [f"r{i}" for i in range(14000)], "text": "safe", "language": "pt", "label": "BENIGN"})
            te = pd.DataFrame({"id": [f"t{i}" for i in range(6000)], "text": "prompt", "language": "pt"})
            sample = pd.DataFrame({"id": te.id, "label": "BENIGN"})
            for name, frame in (("train", tr), ("test", te), ("sample_submission", sample)):
                frame.to_csv(path / f"{name}.csv", index=False)
            sample.to_csv(path / "submission.csv", index=False)
            self.assertTrue(validate_submission(path / "submission.csv", path)["valid"])
            bad = sample.copy()
            bad.loc[0, "id"] = bad.loc[1, "id"]
            bad.to_csv(path / "submission.csv", index=False)
            with self.assertRaises(ValueError):
                validate_submission(path / "submission.csv", path)
            sample.to_csv(path / "submission.csv", index=True)
            with self.assertRaises(ValueError):
                validate_submission(path / "submission.csv", path)


if __name__ == "__main__":
    unittest.main()
