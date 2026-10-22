import argparse
import json
from pathlib import Path

from .data import audit_data, validate_submission, write_json


def main():
    parser = argparse.ArgumentParser(description="RMIT 2026 approved-encoder pipeline")
    sub = parser.add_subparsers(dest="command", required=True)
    audit = sub.add_parser("audit")
    audit.add_argument("--data", required=True)
    audit.add_argument("--output", default="artifacts/audit")
    train = sub.add_parser("train")
    train.add_argument("--data", required=True)
    train.add_argument("--split", default="artifacts/audit/split.csv")
    train.add_argument("--config", required=True)
    train.add_argument("--model-path", required=True)
    train.add_argument("--output", required=True)
    final = sub.add_parser("finalize")
    final.add_argument("--data", required=True)
    final.add_argument("--experiment", required=True)
    final.add_argument("--model-path", required=True)
    final.add_argument("--output", default="artifacts/final")
    final.add_argument("--bias")
    predict = sub.add_parser("predict")
    predict.add_argument("--data", required=True)
    predict.add_argument("--checkpoint", required=True)
    predict.add_argument("--output", required=True)
    predict.add_argument("--bias")
    cal = sub.add_parser("calibrate")
    cal.add_argument("--experiment", required=True)
    validate = sub.add_parser("validate")
    validate.add_argument("--data", required=True)
    validate.add_argument("--submission", required=True)
    validate.add_argument("--output")
    args = parser.parse_args()
    if args.command == "audit":
        audit_data(args.data, args.output)
    elif args.command == "train":
        from .training import train_experiment
        train_experiment(args.data, args.split, args.config, args.model_path, args.output)
    elif args.command == "finalize":
        from .training import finalize
        finalize(args.data, args.experiment, args.model_path, args.output, args.bias)
    elif args.command == "predict":
        from .training import predict as run_predict
        run_predict(args.data, args.checkpoint, args.output, args.bias)
    elif args.command == "calibrate":
        from .calibration import calibrate
        calibrate(args.experiment)
    else:
        result = validate_submission(args.submission, args.data)
        if args.output:
            write_json(args.output, result)
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
