# AGENTS.md

Agent instructions for **sitecheck**, in the cross-tool [AGENTS.md](https://agents.md)
format. Claude Code reads `CLAUDE.md`, which is a **symlink to this file**, so there is
a single source of truth for every tool. Humans: see [README.md](README.md).

## What this repo is

A confounding auditor for medical ML. Given features, a binary label, and which
hospital each row came from, it decides whether a reported result reflects biology or
laboratory provenance. Three probes (site recoverability, label–site association, split
sensitivity) feed one verdict: `CLEAN`, `SITE_SIGNAL_UNEXPLOITED`, `INFLATED`,
`CONFOUNDED`.

## Driving the CLI as an agent

Every command is **non-interactive**, takes **`--json`**, and uses **load-bearing exit
codes** — so any agent can drive it without a TTY or screen-scraping.

```bash
sitecheck doctor --json
# -> {"ok": true, "checks": {...}, "version": "0.1.0"}          exit 0 when ready

sitecheck demo --json
# -> {"ok": true, "cases": {"clean": {...}, ...}}               exit 2 if a verdict regressed

sitecheck audit cohort.parquet --features embedding --layer -1 \
  --label grade --positive G3 --tcga-barcode filename --json
# -> {"ok": true, "verdict": "INFLATED", "trustworthy": false, ...}
```

**Contract:** with `--json`, stdout is exactly one JSON object (success *or* failure:
`{"ok": false, "error": "..."}`). Exit `0` success, `1` error, `2` untrustworthy verdict
under `--strict`. NaN is serialised as `null` — never as bare `NaN`, which is not JSON.

## Setup

```bash
uv sync --extra dev
uv run sitecheck doctor --json
```

## Hard rules (when editing this repo)

1. **CLI-first.** Every capability ships as a `sitecheck` subcommand. No notebook-only
   flows.
2. **`--json` on every command.** Machine-readable output is a contract.
3. **Exit codes mean something.** Never swallow errors; failures go through
   `CliError` → `output.fail`, so agents get JSON on the error path too.
4. **Report a baseline with every metric.** A site probe without its chance rate, or a
   split delta without a CI, is not a finding. This is the whole point of the tool —
   don't violate it in the tool's own output.
5. **Every verdict change needs a synthetic cohort that pins it.** The builders in
   `src/sitecheck/synthetic.py` back both `pytest` and `sitecheck demo`, so documented
   behaviour and tested behaviour cannot drift. Adding a verdict means adding a builder.
6. **Statistical thresholds are named constants with a cited reason.** See
   `MIN_MATERIAL_DROP` in `audit.py`. Don't inline magic numbers — an unexplained
   threshold in a tool that judges other people's rigour is indefensible.
7. **Never coerce missing labels.** `extract_labels` returns a validity mask; folding
   NaN into the negative class invents controls.

## Layout

```
src/sitecheck/audit.py      the three probes + the decision rule
src/sitecheck/synthetic.py  cohorts with known-correct verdicts (tests + demo)
src/sitecheck/loading.py    table → (x, y, site, patient), incl. TCGA barcodes
src/sitecheck/cli.py        typer app
src/sitecheck/output.py     the --json / exit-code contract
scripts/tcga_case_study.py  reproduces the README table on real public data
```

## Build / test / verify

```bash
uv run ruff check src tests scripts
uv run ruff format --check src tests scripts
uv run pytest                             # 21 tests, ~45s
uv run sitecheck demo --json              # verdict regression check
uv run python scripts/tcga_case_study.py  # downloads 445 MB, ~30s after that
```

## Gotchas

- **pyarrow nested columns.** A parquet column of vectors comes back as an object array
  *of* arrays; `np.stack` refuses it. Use `loading._as_float_array`, which round-trips
  through `.tolist()`. This bit once already.
- **Prov-GigaPath embeddings are `(14, 768)` per slide** — one vector per transformer
  layer, not a 14-slide bag. Index with `--layer -1`.
- **`typer` and ruff B008.** Ruff flags calls in argument defaults when the annotation
  is mutable (`Path`, `list[str]`) but not when it's `str`/`int`/`bool`. Annotate paths
  as `str` and convert inside, as `cli.py` does.
- **Bootstrap CIs tighten with n.** Statistical significance alone is not a finding at
  cohort scale; that's why `is_material_drop` also requires an effect size.
