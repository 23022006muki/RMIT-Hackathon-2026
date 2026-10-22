"""Keep a local, convenient bundle; never uploads or includes model binaries."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from protector.data import sha256, validate_submission, write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="../data")
    parser.add_argument("--output", default="../deliverables")
    args = parser.parse_args()
    final = ROOT / "artifacts/final"
    if not (final / "submission_validation.json").exists():
        raise RuntimeError("Finish and validate the actual final model before packaging")
    origin = json.loads((final / "run.json").read_text(encoding="utf-8"))
    if origin.get("execution", {}).get("platform") != "kaggle" or "total_seconds" not in origin:
        raise RuntimeError("Final checkpoint must come from a completed Kaggle training run")
    checkpoint = final / "checkpoint"
    for name in ("model.safetensors", "config.json", "protector_config.json", "training_origin.json", "tokenizer_config.json"):
        if not (checkpoint / name).is_file():
            raise RuntimeError(f"Download the complete final Kaggle checkpoint before packaging: missing {name}")
    checkpoint_origin = json.loads((checkpoint / "training_origin.json").read_text(encoding="utf-8"))
    if checkpoint_origin.get("platform") != "kaggle":
        raise RuntimeError("Final checkpoint origin is not Kaggle")
    result = validate_submission(final / "submission.csv", ROOT / args.data)
    output = (ROOT / args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    status_path = ROOT / "artifacts/submission_status.json"
    submission_status = json.loads(status_path.read_text(encoding="utf-8")) if status_path.exists() else {}
    copies = {
        "submission.csv": final / "submission.csv",
        "UEL_SGS_LucViHoangDe_C1.ipynb": ROOT / "notebooks/UEL_SGS_LucViHoangDe_C1.ipynb",
        "RMIT_SGS_LucViHoangDe_C1.html": ROOT / "reports/results.html",
        "submission_status.json": status_path,
        "RESULTS.md": ROOT / "reports/RESULTS.md",
        "validation.png": ROOT / "reports/validation.png",
        "experiments.csv": ROOT / "reports/experiments.csv",
        "submission_validation.json": final / "submission_validation.json",
    }
    for name, source in copies.items():
        shutil.copy2(source, output / name)
    manifest = {"repository": "https://github.com/23022006muki/RMIT-Hackathon-2026",
                "local_repository": str(ROOT), "data_directory": str((ROOT / args.data).resolve()),
                "final_checkpoint": str((final / "checkpoint").resolve()),
                "pretrained_model_cache": str((ROOT.parent / "cache/kagglehub").resolve()),
                "final_checkpoint_files": {p.name: {"sha256": sha256(p), "bytes": p.stat().st_size} for p in checkpoint.iterdir() if p.is_file()},
                "submission": result, "kaggle_submission_status": submission_status,
                "bundle_files": {name: {"sha256": sha256(output / name), "bytes": (output / name).stat().st_size} for name in copies}}
    write_json(output / "manifest.json", manifest)
    notes = f"""# RMIT_SGS_LucViHoangDe_C1

Mở `RMIT_SGS_LucViHoangDe_C1.html` để xem báo cáo và `UEL_SGS_LucViHoangDe_C1.ipynb` để xem notebook Kaggle.

`submission.csv` có 6.000 dự đoán thật từ model đã fine-tune và đã qua kiểm tra cấu trúc. Đã nộp {submission_status.get("submission_attempts_by_project", 0)} lượt theo yêu cầu của bạn; trạng thái {submission_status.get("status", "unknown")}, public score {submission_status.get("public_score", "unknown")}.

Mã nguồn trên máy: {ROOT}
Dữ liệu gốc: {manifest['data_directory']}
Model cuối: {manifest['final_checkpoint']}
Model pretrained từ Kaggle: {manifest['pretrained_model_cache']}

ZIP chứa mã, notebook, báo cáo và dự đoán; các trọng số lớn và dữ liệu gốc vẫn nằm trong các thư mục trên. Kết quả validation thuộc model chọn cấu hình; model cuối huấn luyện trên toàn bộ train nên điểm validation và public leaderboard được ghi riêng; điểm private leaderboard chưa biết.

Rules đã được đọc qua API Kaggle ngày 22/10/2026. Hạn nộp đã xác minh: 18:00 ngày 22/10/2026, giờ Việt Nam. Kiểm tra trạng thái chia sẻ mã trên Kaggle và giấy phép trong docs/competition.md; đọc cập nhật Rules/Discussion trước khi nộp. Notebook không tự submit. Một lượt CSV đã được nộp theo yêu cầu trực tiếp của bạn; không dùng điểm leaderboard để tối ưu lại model.
"""
    (output / "START_HERE.md").write_text(notes, encoding="utf-8")
    tracked = subprocess.check_output(["git", "ls-files", "-z"], cwd=ROOT).decode().split("\0")
    archive = output / "RMIT_SGS_LucViHoangDe_C1.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for relative in tracked:
            if relative and (ROOT / relative).is_file():
                z.write(ROOT / relative, "project/" + relative)
        for file in list(copies) + ["manifest.json", "START_HERE.md"]:
            z.write(output / file, file)
    print(json.dumps({"directory": str(output), "zip": str(archive), "zip_sha256": sha256(archive)}, indent=2))


if __name__ == "__main__":
    main()
