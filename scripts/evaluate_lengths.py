"""Run validation diagnostics inside Kaggle; never train locally."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from protector.diagnostics import evaluate_lengths

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--split", default="artifacts/audit/split.csv")
    parser.add_argument("--output", default="artifacts/kaggle_length_ablation")
    args = parser.parse_args()
    evaluate_lengths(args.experiment, args.data, args.split, args.output)
