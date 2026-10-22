import hashlib
import json
import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from .constants import LABELS, LANGUAGES


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_data(directory, expected_sizes=True):
    directory = Path(directory)
    frames = {}
    expected = {
        "train": ["id", "text", "language", "label"],
        "test": ["id", "text", "language"],
        "sample_submission": ["id", "label"],
    }
    for name, columns in expected.items():
        frame = pd.read_csv(directory / f"{name}.csv", dtype=str, keep_default_na=False)
        if frame.columns.tolist() != columns:
            raise ValueError(f"{name}: expected columns {columns}, found {frame.columns.tolist()}")
        if frame.id.duplicated().any() or frame.id.str.strip().eq("").any():
            raise ValueError(f"{name}: blank or duplicate ID")
        if frame.apply(lambda s: s.str.strip().eq("")).any().any():
            raise ValueError(f"{name}: empty field")
        if "language" in frame and not set(frame.language) <= set(LANGUAGES):
            raise ValueError(f"{name}: unknown language")
        if "label" in frame and not set(frame.label) <= set(LABELS):
            raise ValueError(f"{name}: unknown label")
        frames[name] = frame
    tr, te, sample = (frames[n] for n in expected)
    if expected_sizes and (len(tr), len(te), len(sample)) != (14000, 6000, 6000):
        raise ValueError("Expected 14,000 train and 6,000 test/sample rows")
    if set(tr.id) & set(te.id):
        raise ValueError("Train/test ID overlap")
    if set(te.id) != set(sample.id):
        raise ValueError("Test/sample IDs differ")
    return tr, te, sample


def normalize_for_group(text):
    # Used only for duplicate grouping; model inputs retain their original text.
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", text).casefold()).strip()


def shingle_set(text):
    words = re.findall(r"\w+", normalize_for_group(text))
    return {" ".join(words[i:i + 3]) for i in range(max(0, len(words) - 2))}


def duplicate_groups(frame, threshold=0.90):
    """Group normalized duplicates and verified MinHash-LSH candidate matches.

    Approximate candidate retrieval can miss matches; all reported matches are
    verified using exact shingle Jaccard. Labels never enter similarity search.
    """
    parent = list(range(len(frame)))

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(a, b):
        parent[root(a)] = root(b)

    normalized = {}
    pairs, shingles = [], []
    # 16 bands x 4 minima; NumPy implementation needs no extra Kaggle package.
    rng = np.random.RandomState(42)
    a = rng.randint(1, 2**32 - 1, size=64, dtype=np.uint64)
    b = rng.randint(0, 2**32 - 1, size=64, dtype=np.uint64)
    buckets = {}
    for i, text in enumerate(frame.text):
        norm = normalize_for_group(text)
        if norm in normalized:
            j = normalized[norm]
            union(i, j)
            pairs.append((i, j, 1.0, "normalized_exact"))
        else:
            normalized[norm] = i
        tokens = shingle_set(text)
        shingles.append(tokens)
        # Very short phrases can match by chance; normalized exact handles them.
        if len(tokens) < 8:
            continue
        hashed = np.array([int.from_bytes(hashlib.blake2b(s.encode("utf-8"), digest_size=4).digest(), "little") for s in sorted(tokens)], dtype=np.uint64)
        minima = ((a[:, None] * hashed + b[:, None]) % np.uint64(4294967311)).min(axis=1)
        keys = [(band, minima[band * 4:(band + 1) * 4].tobytes()) for band in range(16)]
        candidates = set()
        for key in keys:
            candidates.update(buckets.get(key, ()))
        for j in sorted(candidates):
            union_set = tokens | shingles[j]
            similarity = len(tokens & shingles[j]) / len(union_set)
            if similarity >= threshold:
                union(i, j)
                pairs.append((i, j, similarity, "shingle_jaccard"))
        for key in keys:
            buckets.setdefault(key, []).append(i)
    group = np.array([root(i) for i in range(len(frame))])
    pair_frame = pd.DataFrame(pairs, columns=["row_a", "row_b", "similarity", "method"])
    if len(pair_frame):
        pair_frame["id_a"] = pair_frame.row_a.map(frame.id)
        pair_frame["id_b"] = pair_frame.row_b.map(frame.id)
        pair_frame = pair_frame.drop_duplicates(["id_a", "id_b", "method"])
    return group, pair_frame


def audit_data(directory, output):
    tr, te, _ = read_data(directory)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    groups, pairs = duplicate_groups(tr)
    cross_frame = pd.concat([tr[["id", "text"]], te[["id", "text"]]], ignore_index=True)
    _, all_pairs = duplicate_groups(cross_frame)
    cross_pairs = all_pairs[(all_pairs.row_a >= len(tr)) & (all_pairs.row_b < len(tr))] if len(all_pairs) else all_pairs
    pairs.to_csv(output / "near_duplicates_train.csv", index=False)
    cross_pairs.to_csv(output / "near_duplicates_train_test.csv", index=False)
    split = make_split(tr, groups, seed=42)
    split.to_csv(output / "split.csv", index=False)
    stats = {
        "rows": {"train": len(tr), "test": len(te)},
        "file_sha256": {name: sha256(Path(directory) / name) for name in ("train.csv", "test.csv", "sample_submission.csv")},
        "labels": tr.label.value_counts().to_dict(),
        "train_languages": tr.language.value_counts().to_dict(),
        "test_languages": te.language.value_counts().to_dict(),
        "label_by_language": pd.crosstab(tr.language, tr.label).to_dict(),
        "length_characters": {},
        "exact_train_test_text_overlap": len(set(tr.text) & set(te.text)),
        "normalized_train_test_overlap": len(set(tr.text.map(normalize_for_group)) & set(te.text.map(normalize_for_group))),
        "near_duplicate_train_pairs": len(pairs),
        "near_duplicate_cross_pairs": len(cross_pairs),
        "group_count": int(len(set(groups))),
        "duplicate_method": "Normalized NFKC/casefold/whitespace; deterministic 64-minimum MinHash LSH (16 bands x 4), exact word-trigram Jaccard >= .90, >=8 shingles; approximate search may miss matches.",
        "split_counts": split.partition.value_counts().to_dict(),
    }
    for name, frame in (("train", tr), ("test", te)):
        lengths = frame.text.str.len()
        stats["length_characters"][name] = {str(q): float(lengths.quantile(q)) for q in (0, .5, .9, .95, .99, 1)}
    write_json(output / "data_audit.json", stats)
    lines = ["# Data audit", "", "Counts and hashes are computed from local competition files. No prompt text is published.", "", "```json", json.dumps(stats, ensure_ascii=False, indent=2), "```", "", "Validation split uses StratifiedGroupKFold on language × label, grouping detected near duplicates. Cross-split groups are forbidden. Train/test near matches are reported, never used to infer test labels."]
    (output / "data_audit.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(stats, indent=2), flush=True)
    return stats


def make_split(frame, groups, seed=42):
    strata = frame.language + "_" + frame.label
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    train_idx, val_idx = next(splitter.split(frame, strata, groups))
    if set(groups[train_idx]) & set(groups[val_idx]):
        raise AssertionError("Group leakage")
    result = frame[["id"]].copy()
    result["group"] = groups
    result["partition"] = "train"
    result.loc[val_idx, "partition"] = "validation"
    return result


def load_split(frame, path):
    split = pd.read_csv(path, dtype={"id": str})
    if split.id.duplicated().any() or set(split.id) != set(frame.id):
        raise ValueError("Split does not match train IDs")
    merged = frame.merge(split, on="id", validate="one_to_one", sort=False)
    if set(merged.partition) != {"train", "validation"}:
        raise ValueError("Split requires train and validation")
    train = merged[merged.partition == "train"].reset_index(drop=True)
    val = merged[merged.partition == "validation"].reset_index(drop=True)
    if set(train.group) & set(val.group):
        raise ValueError("Group leakage in saved split")
    return train, val


def validate_submission(path, directory):
    _, te, sample = read_data(directory)
    sub = pd.read_csv(path, dtype=str, keep_default_na=False)
    if sub.columns.tolist() != ["id", "label"]:
        raise ValueError("Submission columns must be exactly id,label")
    if len(sub) != 6000 or sub.id.duplicated().any() or set(sub.id) != set(te.id):
        raise ValueError("Submission requires all 6,000 test IDs exactly once")
    if not sub.label.isin(LABELS).all():
        raise ValueError("Invalid or empty submission label")
    if sub.id.tolist() != sample.id.tolist():
        raise ValueError("Submission order differs from sample_submission.csv")
    result = {"valid": True, "rows": len(sub), "columns": sub.columns.tolist(), "label_counts": sub.label.value_counts().to_dict(), "sha256": sha256(path), "kaggle_submissions_used_by_this_project": 0}
    return result
