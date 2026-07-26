"""Audit real TCGA cohorts and print the markdown table in the README.

Uses public Prov-GigaPath slide embeddings (445 MB, one parquet, no GPU):
https://huggingface.co/datasets/seandavis/tcga_provgigapath_embeddings

    uv run python scripts/tcga_case_study.py [--data PATH] [--json]
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

from sitecheck import audit

URL = (
    "https://huggingface.co/datasets/seandavis/tcga_provgigapath_embeddings/"
    "resolve/main/provgigapath_embeddings_with_metadata.parquet"
)
DEFAULT_PATH = Path("data/tcga_provgigapath.parquet")

# Grade strings vary by cohort; G3/G4 is the conventional "high grade" cut.
HIGH_GRADE = {"G3": 1, "G4": 1, "G1": 0, "G2": 0}
TWO_YEARS_IN_DAYS = 730

GRADE_COHORTS = ["KIRC", "HNSC", "LIHC", "STAD"]
SURVIVAL_COHORTS = ["KIRC", "GBM", "LGG", "HNSC", "UCEC", "LUAD"]


def load(path: Path) -> pd.DataFrame:
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        print(f"downloading 445 MB -> {path}", file=sys.stderr)
        urllib.request.urlretrieve(URL, path)  # noqa: S310 - constant https URL
    df = pd.read_parquet(path)
    # Prov-GigaPath emits one vector per transformer layer; the last is the slide
    # representation people actually use.
    df["x"] = [np.asarray(e[-1], dtype=np.float64) for e in df["embedding"]]
    barcode = df["filename"].str.extract(r"TCGA-([0-9A-Z]{2})-([0-9A-Z]{4})")
    df["site"] = barcode[0]
    df["patient"] = "TCGA-" + barcode[0] + "-" + barcode[1]
    return df


def _audit(sub: pd.DataFrame, y: np.ndarray, name: str) -> dict | None:
    if len(sub) < 50 or y.sum() < 20 or (1 - y).sum() < 20:
        return None
    rep = audit(np.stack(sub["x"].values), y, sub["site"].values, sub["patient"].values)
    s, a = rep.split_sensitivity, rep.label_site_association
    return {
        "task": name,
        "n": rep.n,
        "sites": s.n_sites_evaluated,
        "site_only_auc": a.site_only_auc,
        "leakage": rep.site_recoverability.leakage_score,
        "random_auc": s.random_auc,
        "site_out_auc": s.site_out_auc,
        "delta": s.delta,
        "ci": s.delta_ci95,
        "verdict": rep.verdict,
    }


def grade_task(df: pd.DataFrame, cohort: str) -> dict | None:
    m = (df["cancer type abbreviation"] == cohort) & df["histological_grade"].isin(HIGH_GRADE)
    sub = df[m]
    y = sub["histological_grade"].map(HIGH_GRADE).to_numpy(dtype=int)
    return _audit(sub, y, f"{cohort} · high grade")


def survival_task(df: pd.DataFrame, cohort: str) -> dict | None:
    m = (df["cancer type abbreviation"] == cohort) & df["OS"].notna() & df["OS.time"].notna()
    sub = df[m]
    time, event = sub["OS.time"].to_numpy(), sub["OS"].to_numpy()
    # Drop anyone censored before 2 years: their 2-year outcome is unknown, and
    # scoring them as survivors would fabricate the label.
    known = (time >= TWO_YEARS_IN_DAYS) | (event == 1)
    sub = sub[known]
    y = ((sub["OS.time"].to_numpy() < TWO_YEARS_IN_DAYS) & (sub["OS"].to_numpy() == 1)).astype(int)
    return _audit(sub, y, f"{cohort} · died within 2y")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=Path, default=DEFAULT_PATH)
    ap.add_argument("--json", action="store_true", dest="as_json")
    args = ap.parse_args()

    df = load(args.data)
    print(
        f"{len(df)} slides · {df['patient'].nunique()} patients · "
        f"{df['site'].nunique()} sites · {df['cancer type abbreviation'].nunique()} cancer types",
        file=sys.stderr,
    )

    rows = [grade_task(df, c) for c in GRADE_COHORTS]
    rows += [survival_task(df, c) for c in SURVIVAL_COHORTS]
    rows = [r for r in rows if r]

    if args.as_json:
        print(json.dumps(rows, indent=2, default=float))
        return 0

    print(
        "| task | n | sites | site-only AUC | leakage | random AUC | unseen-site AUC "
        "| delta (95% CI) | verdict |"
    )
    print("|---|--:|--:|--:|--:|--:|--:|:--|---|")
    for r in rows:
        print(
            f"| {r['task']} | {r['n']} | {r['sites']} | {r['site_only_auc']:.3f} "
            f"| {r['leakage']:.2f} | {r['random_auc']:.3f} | {r['site_out_auc']:.3f} "
            f"| {r['delta']:+.3f} ({r['ci'][0]:+.3f}, {r['ci'][1]:+.3f}) | {r['verdict']} |"
        )
    flagged = sum(r["verdict"] in {"INFLATED", "CONFOUNDED"} for r in rows)
    print(f"\n{flagged}/{len(rows)} tasks flagged as untrustworthy under a site-aware split.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
