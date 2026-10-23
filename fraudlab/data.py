"""Validate the original Kaggle data and record deliberate masking separately."""
from pathlib import Path
import hashlib
import json
import tempfile
import urllib.request
import zipfile
import numpy as np
import pandas as pd

INPUTS = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]
COLUMNS = INPUTS + ["Class"]
SOURCE = "https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud"
DOWNLOAD = "https://www.kaggle.com/api/v1/datasets/download/mlg-ulb/creditcardfraud"
VERIFIED_KAGGLE_SHA256 = "76274b691b16a6c49d3f159c883398e03ccd6d1ee12d9d8ee38f4b4b98551a89"


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def validate(frame, labeled=True):
    required = COLUMNS if labeled else INPUTS
    if set(frame) != set(required):
        raise ValueError(f"Expected exactly {required}; got {list(frame)}")
    x = frame[required].apply(pd.to_numeric, errors="raise")
    if x.empty or np.isinf(x.to_numpy()).any():
        raise ValueError("Empty data or infinite values")
    if x.Time.isna().any() or (x.Time < 0).any():
        raise ValueError("Time must be observed and nonnegative")
    if (x.Amount.dropna() < 0).any():
        raise ValueError("Negative transaction amount")
    if labeled and (x.Class.isna().any() or not x.Class.isin([0, 1]).all()):
        raise ValueError("Class must be observed binary labels")
    return x


def download(destination):
    """Public Kaggle ZIP only. Fail clearly; never substitute generated records."""
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        archive = Path(tmp) / "source.zip"
        request = urllib.request.Request(DOWNLOAD, headers={"User-Agent": "fraudlab/1.0"})
        try:
            with urllib.request.urlopen(request, timeout=90) as r, archive.open("wb") as f:
                while block := r.read(1024 * 1024):
                    f.write(block)
            with zipfile.ZipFile(archive) as z:
                # Read this one member, never extract arbitrary archive paths.
                with z.open("creditcard.csv") as source, destination.open("wb") as target:
                    import shutil
                    shutil.copyfileobj(source, target)
        except Exception as e:
            raise RuntimeError(
                f"Kaggle download failed. Download creditcard.csv manually from {SOURCE} "
                f"and pass its path; no synthetic fallback. Cause: {e}"
            ) from e
    frame, report = load(destination, deduplicate=False)
    if (report["raw_rows"] != 284807 or int(frame.Class.sum()) != 492
            or report["file_sha256"] != VERIFIED_KAGGLE_SHA256):
        destination.unlink()
        raise ValueError("Downloaded data does not match the verified Kaggle CSV bytes/counts")
    write_json(destination.with_suffix(".source.json"), report)
    return report


def load(path, deduplicate=True):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Missing real CSV: {path}. Run fraudlab download.")
    frame = validate(pd.read_csv(path))
    frame["Class"] = frame.Class.astype(int)
    # Same feature vector with conflicting targets is not silently resolved.
    feature_duplicates = frame.duplicated(INPUTS, keep=False)
    if feature_duplicates.any():
        conflicts = frame.loc[feature_duplicates].groupby(INPUTS, dropna=False).Class.nunique()
        if (conflicts > 1).any():
            raise ValueError("Identical input features have conflicting labels")
    duplicate_count = int(frame.duplicated().sum())
    raw_rows = len(frame)
    raw_counts = {str(k): int(v) for k, v in frame.Class.value_counts().items()}
    raw_missing = int(frame[INPUTS].isna().sum().sum())
    # Only byte-value-identical rows removed. Do not conflate same timestamps with duplicates.
    frame["source_row"] = np.arange(raw_rows)
    if deduplicate:
        frame = frame.drop_duplicates(COLUMNS)
    frame = frame.sort_values(["Time", "source_row"], kind="stable").reset_index(drop=True)
    report = {
        "source_url": SOURCE, "file_sha256": sha256(path), "file_bytes": path.stat().st_size,
        "raw_rows": raw_rows, "raw_class_counts": raw_counts,
        "raw_missing_input_cells": raw_missing,
        "exact_duplicate_rows": duplicate_count, "deduplicate": deduplicate,
        "used_rows": len(frame), "used_fraud_rows": int(frame.Class.sum()),
        "columns": COLUMNS, "time_range_seconds": [float(frame.Time.min()), float(frame.Time.max())],
        "limitations": ["No card/customer IDs", "V1-V28 business meanings undisclosed",
                        "Time is elapsed time, not verified local clock time", "Two-day historical sample"],
    }
    return frame, report


def chronological_parts(frame, train_fraction=.6, validation_fraction=.2):
    if not (0 < train_fraction < 1 and 0 < validation_fraction < 1 - train_fraction):
        raise ValueError("Invalid chronological split fractions")
    times = frame.Time.to_numpy()
    if np.any(np.diff(times) < 0):
        raise ValueError("Sort by Time before splitting")
    # Keep tied timestamps in the same outer partition.
    a = int(np.searchsorted(times, times[int(len(frame) * train_fraction)], side="left"))
    b = int(np.searchsorted(times, times[int(len(frame) * (train_fraction + validation_fraction))], side="left"))
    parts = [frame.iloc[:a].copy(), frame.iloc[a:b].copy(), frame.iloc[b:].copy()]
    for name, part in zip(["train", "validation", "test"], parts):
        if part.empty or part.Class.nunique() != 2:
            raise ValueError(f"{name} requires both classes; change the split or supply more data")
    return parts


def inject_missing(parts, count, seed=42):
    """Exactly count previously observed cells, MCAR, never Class or Time.

    Allocation uses partition sizes, not labels. Each partition gets its own RNG.
    Returns masked partitions and source-row/column manifest for every injected cell.
    """
    if count < 0:
        raise ValueError("missing_cells must be nonnegative")
    columns = INPUTS[1:]
    capacities = np.array([p[columns].notna().to_numpy().sum() for p in parts], dtype=int)
    if count > capacities.sum():
        raise ValueError("Requested more missing cells than observed eligible cells")
    exact = count * capacities / max(1, capacities.sum())
    allocation = np.floor(exact).astype(int)
    for i in np.argsort(-(exact - allocation))[:count - int(allocation.sum())]:
        allocation[i] += 1
    output, manifests = [], []
    for i, (part, number) in enumerate(zip(parts, allocation)):
        part = part.copy()
        values = part[columns].to_numpy(copy=True)
        eligible = np.flatnonzero(~np.isnan(values.ravel()))
        chosen = np.random.default_rng(seed + i).choice(eligible, int(number), replace=False)
        row, col = np.unravel_index(chosen, values.shape)
        manifest = pd.DataFrame({"source_row": part.source_row.to_numpy()[row],
                                 "column": np.array(columns)[col]})
        values[row, col] = np.nan
        part.loc[:, columns] = values
        output.append(part)
        manifests.append(manifest)
    return output, manifests
