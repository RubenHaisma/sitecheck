"""Turn a tabular file into the four arrays :func:`sitecheck.audit` needs."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from sitecheck.output import CliError

# TCGA barcodes look like TCGA-06-0138-01Z-00-DX2...: the 2nd field is the tissue
# source site (the submitting hospital) and the 3rd identifies the participant.
_TCGA_BARCODE = re.compile(r"TCGA-([0-9A-Z]{2})-([0-9A-Z]{4})")


def read_table(path: Path, columns: list[str] | None = None) -> pd.DataFrame:
    if not path.exists():
        raise CliError(f"no such file: {path}")
    suffix = path.suffix.lower()
    try:
        if suffix == ".parquet":
            return pd.read_parquet(path, columns=columns)
        if suffix in {".csv", ".tsv"}:
            return pd.read_csv(path, sep="\t" if suffix == ".tsv" else ",", usecols=columns)
    except Exception as exc:  # narrow types vary by engine; surface the message
        raise CliError(f"could not read {path}: {exc}") from exc
    raise CliError(f"unsupported file type '{suffix}' (use .parquet, .csv or .tsv)")


def _as_float_array(value: object) -> np.ndarray:
    """Materialise one cell as a float array.

    pyarrow hands nested list columns back as an object array *of* arrays, which
    ``np.asarray`` leaves as dtype=object and ``np.stack`` then refuses. Round-tripping
    through ``.tolist()`` gives numpy real nested lists it can type.
    """
    if isinstance(value, np.ndarray) and value.dtype == object:
        return np.array(value.tolist(), dtype=np.float64)
    arr = np.asarray(value)
    if arr.dtype == object:
        return np.array(arr.tolist(), dtype=np.float64)
    return arr.astype(np.float64, copy=False)


def _require(df: pd.DataFrame, col: str, what: str) -> None:
    if col not in df.columns:
        near = [c for c in df.columns if col.lower() in str(c).lower()][:5]
        hint = f" Did you mean one of {near}?" if near else ""
        raise CliError(f"{what} column '{col}' not in the table.{hint}")


def extract_features(df: pd.DataFrame, spec: str, layer: int | None) -> np.ndarray:
    """Build the feature matrix.

    ``spec`` is either one column holding a vector per row, or a comma-separated list
    of numeric columns. ``layer`` indexes into a per-layer embedding such as
    Prov-GigaPath's ``(14, 768)``.
    """
    names = [s.strip() for s in spec.split(",") if s.strip()]
    if len(names) > 1:
        for n in names:
            _require(df, n, "feature")
        return df[names].to_numpy(dtype=np.float64)

    col = names[0]
    _require(df, col, "feature")
    values = df[col].to_numpy()
    first = values[0]
    if not hasattr(first, "__len__"):
        raise CliError(
            f"column '{col}' holds scalars, not vectors. Pass several numeric columns "
            f"as --features a,b,c, or point at a column of embeddings."
        )

    arr0 = _as_float_array(first)
    if arr0.ndim == 2:
        if layer is None:
            raise CliError(
                f"column '{col}' holds a {arr0.shape} matrix per row (a per-layer "
                f"embedding). Choose one with --layer (e.g. --layer -1 for the last)."
            )
        if not -arr0.shape[0] <= layer < arr0.shape[0]:
            raise CliError(f"--layer {layer} out of range for shape {arr0.shape}")
        return np.stack([_as_float_array(v)[layer] for v in values])
    if arr0.ndim != 1:
        raise CliError(f"column '{col}' has unsupported per-row shape {arr0.shape}")

    widths = {_as_float_array(v).shape for v in values[:200]}
    if len(widths) > 1:
        raise CliError(f"column '{col}' has ragged vectors (shapes seen: {sorted(widths)})")
    return np.stack([_as_float_array(v) for v in values])


def extract_labels(
    df: pd.DataFrame, col: str, positive: str | None
) -> tuple[np.ndarray, np.ndarray]:
    """Binarise the outcome column.

    Returns ``(y, valid)``. Missing labels are reported in ``valid`` rather than being
    coerced — silently folding NaN into the negative class would invent controls.
    """
    _require(df, col, "label")
    s = df[col]
    valid = s.notna().to_numpy()
    values = sorted(str(v) for v in s.dropna().unique())
    if positive is not None:
        if positive not in values:
            raise CliError(f"--positive '{positive}' not among values of '{col}': {values[:10]}")
        return (s.astype(str) == positive).to_numpy().astype(int), valid
    if len(values) != 2:
        raise CliError(
            f"label '{col}' has {len(values)} distinct values {values[:10]}; sitecheck audits "
            f"binary tasks. Pass --positive VALUE to binarise it."
        )
    return (s.astype(str) == values[1]).to_numpy().astype(int), valid


def extract_column(df: pd.DataFrame, col: str, what: str) -> np.ndarray:
    """Fetch one column as an array, failing with a suggestion rather than a KeyError."""
    _require(df, col, what)
    return df[col].to_numpy()


def extract_barcode_field(df: pd.DataFrame, col: str, which: str) -> np.ndarray:
    """Pull the hospital or participant out of a TCGA barcode column."""
    _require(df, col, "barcode")
    parts = df[col].astype(str).str.extract(_TCGA_BARCODE)
    if parts[0].isna().all():
        raise CliError(
            f"no TCGA barcode found in '{col}' (expected e.g. TCGA-06-0138-01Z-00-DX1...)"
        )
    if which == "site":
        return parts[0].to_numpy()
    return ("TCGA-" + parts[0] + "-" + parts[1]).to_numpy()


def apply_filters(df: pd.DataFrame, filters: list[str]) -> pd.DataFrame:
    """Subset rows with repeated ``col=value`` flags, e.g. one cancer type."""
    for f in filters:
        if "=" not in f:
            raise CliError(f"--filter must look like COLUMN=VALUE; got '{f}'")
        col, _, want = f.partition("=")
        col, want = col.strip(), want.strip()
        _require(df, col, "filter")
        df = df[df[col].astype(str) == want]
        if df.empty:
            raise CliError(f"filter '{f}' left no rows")
    return df
