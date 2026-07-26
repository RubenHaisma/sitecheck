"""sitecheck — the binary. One Typer app, one subcommand per capability.

House rules (enforced in CI):
- every command takes ``--json``
- exit codes are load-bearing (0 ok, 1 error, 2 audit verdict not trustworthy)
- no command writes state outside the path you pass it
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import typer

from sitecheck import __version__
from sitecheck.audit import AuditReport, audit
from sitecheck.loading import (
    apply_filters,
    extract_barcode_field,
    extract_column,
    extract_features,
    extract_labels,
    read_table,
)
from sitecheck.output import EXIT_VERDICT_FAILED, CliError, emit, fail
from sitecheck.synthetic import SCENARIOS

app = typer.Typer(
    name="sitecheck",
    help="Audit whether a medical-ML result measures biology or the lab that made the data.",
    no_args_is_help=True,
    add_completion=False,
)

# Verdicts that mean "do not report the random-split number as your result".
UNTRUSTWORTHY = {"CONFOUNDED", "INFLATED"}
# Below this the leave-one-site-out folds are too small to say anything.
MIN_ROWS = 50

_GLYPH = {
    "CLEAN": "[green]CLEAN[/green]",
    "SITE_SIGNAL_UNEXPLOITED": "[yellow]SITE SIGNAL, UNEXPLOITED[/yellow]",
    "INFLATED": "[red]INFLATED[/red]",
    "CONFOUNDED": "[red]CONFOUNDED[/red]",
}


def _render(rep: AuditReport, title: str) -> str:
    rec, assoc, split = rep.site_recoverability, rep.label_site_association, rep.split_sensitivity
    lines = [
        f"[bold]{title}[/bold]",
        f"  {rep.n} rows · {rep.n_patients} patients · {rep.n_sites} sites · "
        f"prevalence {rep.label_prevalence:.1%}",
        "",
        f"  site recoverable from features   {rec.balanced_accuracy:.3f} balanced acc "
        f"(chance {rec.chance:.3f}) → leakage {rec.leakage_score:.2f}",
        f"  label explained by site alone    AUC {assoc.site_only_auc:.3f} · "
        f"Cramer's V {assoc.cramers_v:.2f}",
        f"  random split (grouped)           AUC {split.random_auc:.3f}",
        f"  unseen site ({split.n_sites_evaluated} held out)"
        f"{'':<9} AUC {split.site_out_auc:.3f}  "
        f"[{split.delta:+.3f}, 95% CI {split.delta_ci95[0]:+.3f} to {split.delta_ci95[1]:+.3f}]",
        "",
        f"  verdict: {_GLYPH.get(rep.verdict, rep.verdict)}",
    ]
    lines += [f"    · {r}" for r in rep.reasons]
    return "\n".join(lines)


def _payload(rep: AuditReport, *, ok: bool = True) -> dict:
    d = rep.to_dict()
    d["ok"] = ok
    d["trustworthy"] = rep.verdict not in UNTRUSTWORTHY
    return d


@app.command("audit")
def audit_file(  # noqa: PLR0913 - each flag maps to one column in the user's table
    data: str = typer.Argument(..., help="Table of features + labels (.parquet/.csv/.tsv)."),
    features: str = typer.Option(
        ..., "--features", help="Embedding column, or comma-separated numeric columns."
    ),
    label: str = typer.Option(..., "--label", help="Binary outcome column."),
    site: str | None = typer.Option(
        None, "--site", help="Column holding hospital / scanner / batch."
    ),
    patient: str | None = typer.Option(None, "--patient", help="Patient identifier column."),
    tcga_barcode: str | None = typer.Option(
        None,
        "--tcga-barcode",
        help="Derive site and patient from a TCGA barcode column instead of --site/--patient.",
    ),
    positive: str | None = typer.Option(
        None, "--positive", help="Value of --label counted as positive."
    ),
    layer: int | None = typer.Option(
        None, "--layer", help="Index into a per-layer embedding, e.g. -1 for the last."
    ),
    filters: list[str] = typer.Option(  # noqa: B008 - typer's API needs the call in the default
        [], "--filter", help="Repeatable COLUMN=VALUE row filter (e.g. cohort=KIRC)."
    ),
    strict: bool = typer.Option(
        False, "--strict", help=f"Exit {EXIT_VERDICT_FAILED} when the verdict is untrustworthy."
    ),
    json_out: bool = typer.Option(False, "--json"),
) -> None:
    """Audit one binary task in a table for site confounding."""
    try:
        if tcga_barcode is None and (site is None or patient is None):
            raise CliError("pass --site and --patient, or --tcga-barcode to derive both")

        path = Path(data)
        df = apply_filters(read_table(path), filters)
        x = extract_features(df, features, layer)
        y, label_valid = extract_labels(df, label, positive)

        if tcga_barcode is not None:
            sites = extract_barcode_field(df, tcga_barcode, "site")
            patients = extract_barcode_field(df, tcga_barcode, "patient")
        else:
            sites = extract_column(df, site, "site")
            patients = extract_column(df, patient, "patient")

        keep = label_valid & ~np.isnan(x).any(axis=1)
        dropped = int((~keep).sum())
        x, y, sites, patients = x[keep], y[keep], sites[keep], patients[keep]
        if len(y) < MIN_ROWS:
            raise CliError(
                f"only {len(y)} usable rows after dropping {dropped} with a missing label or "
                f"feature; need at least {MIN_ROWS}"
            )

        rep = audit(x, y, sites, patients)
    except CliError as err:
        fail(err, json_out=json_out)
        return
    except ValueError as err:
        fail(CliError(str(err)), json_out=json_out)
        return

    title = f"{path.name} · {label}" + (f" · {' '.join(filters)}" if filters else "")
    emit(_payload(rep), json_out=json_out, human=_render(rep, title))
    if strict and rep.verdict in UNTRUSTWORTHY:
        raise typer.Exit(code=EXIT_VERDICT_FAILED)


@app.command()
def demo(json_out: bool = typer.Option(False, "--json")) -> None:
    """Audit four synthetic cohorts whose right answer is known by construction.

    Needs no data — this is the fastest way to see what each verdict looks like, and
    it doubles as a self-test of the decision rule.
    """
    results, blocks, all_ok = {}, [], True
    for name, (builder, expected) in SCENARIOS.items():
        rep = audit(*builder())
        ok = rep.verdict == expected
        all_ok &= ok
        results[name] = {"expected": expected, "got": rep.verdict, "ok": ok, **_payload(rep)}
        blocks.append(_render(rep, f"{name}  (expected {expected})"))

    footer = "all verdicts as expected" if all_ok else "MISMATCH - decision rule regressed"
    emit(
        {"ok": all_ok, "cases": results},
        json_out=json_out,
        human="\n\n".join(blocks) + f"\n\n[bold]{footer}[/bold]",
    )
    if not all_ok:
        raise typer.Exit(code=EXIT_VERDICT_FAILED)


@app.command()
def doctor(json_out: bool = typer.Option(False, "--json")) -> None:
    """Check the environment is ready."""
    checks = {
        name: importlib.util.find_spec(name) is not None
        for name in ("numpy", "pandas", "sklearn", "pyarrow")
    }
    ok = all(checks.values())
    human = "\n".join(f"  {'ok  ' if v else 'MISS'} {k}" for k, v in checks.items())
    emit({"ok": ok, "checks": checks, "version": __version__}, json_out=json_out, human=human)
    if not ok:
        raise typer.Exit(code=1)


@app.command()
def version(json_out: bool = typer.Option(False, "--json")) -> None:
    """Print the sitecheck version."""
    if json_out:
        typer.echo(f'{{"version": "{__version__}"}}')
    else:
        typer.echo(__version__)


if __name__ == "__main__":
    app()
