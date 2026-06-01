"""
Load the California Housing dataset from a stable on-disk CSV path.
The first time this is called, it downloads a small bundled copy of the
8-feature/median-house-value version (Pace & Barry 1997) into ./data.

Returns X (n x 8 float matrix) and y (length-n float vector).
"""
from __future__ import annotations

import csv
import os
import urllib.request

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
CSV_PATH = os.path.join(DATA_DIR, "california_housing.csv")
URL = (
    "https://raw.githubusercontent.com/ageron/handson-ml2/master/"
    "datasets/housing/housing.csv"
)

COLUMNS = [
    "longitude", "latitude", "housing_median_age", "total_rooms",
    "total_bedrooms", "population", "households", "median_income",
    "median_house_value",
]


def _download() -> None:
    os.makedirs(DATA_DIR, exist_ok=True)
    if os.path.exists(CSV_PATH):
        return
    print(f"Downloading California Housing dataset to {CSV_PATH} ...")
    urllib.request.urlretrieve(URL, CSV_PATH)


def _to_float(s: str) -> float:
    s = s.strip()
    if s == "" or s.lower() == "na":
        return float("nan")
    return float(s)


def _row_is_complete(row: list[str]) -> bool:
    return all(cell.strip() != "" and cell.strip().lower() != "na" for cell in row[:8])


def load() -> tuple[list[list[float]], list[float]]:
    """Load the dataset; download on first call.

    Drops rows with any missing feature value, then returns features and
    the median house value target scaled to units of $100,000.
    """
    _download()
    X: list[list[float]] = []
    y: list[float] = []
    with open(CSV_PATH, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            vals = [row[c] for c in COLUMNS]
            if not _row_is_complete(vals):
                continue
            feats = [_to_float(v) for v in vals[:8]]
            target = _to_float(vals[8])
            X.append(feats)
            y.append(target / 100_000.0)  # scale to ~[0.5, 5.0]
    return X, y
