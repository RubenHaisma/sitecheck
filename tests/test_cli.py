"""The CLI contract: one JSON object on stdout, load-bearing exit codes."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
from typer.testing import CliRunner

from sitecheck.cli import app
from sitecheck.loading import extract_features, extract_labels
from sitecheck.output import EXIT_VERDICT_FAILED
from sitecheck.synthetic import make_confounded

runner = CliRunner()


@pytest.fixture
def confounded_parquet(tmp_path):
    """A TCGA-shaped table: one (14, 768)-style per-layer embedding column per row."""
    x, y, site, patient = make_confounded()
    layers = 3
    stacked = [np.tile(row, (layers, 1)) for row in x]
    df = pd.DataFrame(
        {
            "embedding": [s.tolist() for s in stacked],
            "grade": np.where(y == 1, "G3", "G1"),
            "filename": [
                f"TCGA-{s[1:].zfill(2)}-{i:04d}-01Z-00-DX1.h5" for i, s in enumerate(site)
            ],
            "site": site,
            "patient": patient,
            "cohort": "DEMO",
        }
    )
    path = tmp_path / "cohort.parquet"
    df.to_parquet(path)
    return path


def _run(*args):
    return runner.invoke(app, list(args))


def test_doctor_emits_one_json_object():
    res = _run("doctor", "--json")
    assert res.exit_code == 0
    assert json.loads(res.stdout)["ok"] is True


def test_audit_reports_verdict_as_json(confounded_parquet):
    res = _run(
        "audit", str(confounded_parquet),
        "--features", "embedding", "--layer", "-1",
        "--label", "grade", "--positive", "G3",
        "--site", "site", "--patient", "patient",
        "--json",
    )  # fmt: skip
    assert res.exit_code == 0, res.stdout
    payload = json.loads(res.stdout)
    assert payload["verdict"] == "CONFOUNDED"
    assert payload["trustworthy"] is False


def test_strict_exits_nonzero_on_an_untrustworthy_verdict(confounded_parquet):
    """So CI can gate a merge on provenance the way it gates on a failing test."""
    args = [
        "audit", str(confounded_parquet),
        "--features", "embedding", "--layer", "-1",
        "--label", "grade", "--positive", "G3",
        "--site", "site", "--patient", "patient",
        "--json",
    ]  # fmt: skip
    assert _run(*args).exit_code == 0
    assert _run(*args, "--strict").exit_code == EXIT_VERDICT_FAILED


def test_tcga_barcode_derives_site_and_patient(confounded_parquet):
    res = _run(
        "audit", str(confounded_parquet),
        "--features", "embedding", "--layer", "-1",
        "--label", "grade", "--positive", "G3",
        "--tcga-barcode", "filename",
        "--json",
    )  # fmt: skip
    assert res.exit_code == 0, res.stdout
    assert json.loads(res.stdout)["n_sites"] == 8


def test_filter_subsets_rows(confounded_parquet):
    res = _run(
        "audit", str(confounded_parquet),
        "--features", "embedding", "--layer", "-1",
        "--label", "grade", "--positive", "G3",
        "--site", "site", "--patient", "patient",
        "--filter", "cohort=DEMO", "--json",
    )  # fmt: skip
    assert json.loads(res.stdout)["n"] == 560


def test_errors_are_json_when_json_requested(confounded_parquet):
    res = _run(
        "audit", str(confounded_parquet),
        "--features", "embedding", "--layer", "-1",
        "--label", "grade", "--positive", "G3",
        "--site", "nope", "--patient", "patient",
        "--json",
    )  # fmt: skip
    assert res.exit_code == 1
    payload = json.loads(res.stdout)
    assert payload["ok"] is False and "nope" in payload["error"]


def test_missing_layer_on_a_matrix_column_is_explained(confounded_parquet):
    res = _run(
        "audit", str(confounded_parquet),
        "--features", "embedding",
        "--label", "grade", "--positive", "G3",
        "--site", "site", "--patient", "patient",
        "--json",
    )  # fmt: skip
    assert res.exit_code == 1
    assert "--layer" in json.loads(res.stdout)["error"]


def test_demo_all_verdicts_match():
    res = _run("demo", "--json")
    assert res.exit_code == 0, res.stdout
    cases = json.loads(res.stdout)["cases"]
    assert all(c["ok"] for c in cases.values()), {k: v["got"] for k, v in cases.items()}


# ---------------------------------------------------------------- loader units


def test_nested_object_arrays_are_materialised(confounded_parquet):
    """Regression: pyarrow returns nested lists as object arrays of arrays."""
    df = pd.read_parquet(confounded_parquet)
    x = extract_features(df, "embedding", -1)
    assert x.dtype == np.float64
    assert x.shape == (560, 24)


def test_missing_labels_are_reported_not_coerced(tmp_path):
    df = pd.DataFrame({"a": [1.0, 2.0, 3.0], "y": ["G3", None, "G1"]})
    y, valid = extract_labels(df, "y", "G3")
    assert valid.tolist() == [True, False, True]
    assert y[0] == 1 and y[2] == 0
