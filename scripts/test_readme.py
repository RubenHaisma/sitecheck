"""Keep the README honest.

Three checks, none of which need the 445 MB download:

1. Every verdict the code can emit is documented, and vice versa.
2. Every ``sitecheck`` invocation shown in the README parses against the real CLI.
3. The stated thresholds match the constants in ``audit.py``.

Then, only if ``data/tcga_provgigapath.parquet`` is already present, re-run the case
study and diff it against the table in the README.
"""

from __future__ import annotations

import re
import shlex
import subprocess
import sys
from pathlib import Path

from typer.main import get_command
from typer.testing import CliRunner

from sitecheck.audit import MIN_MATERIAL_DROP, MIN_SITE_N, SITE_ONLY_AUC_CONFOUNDED
from sitecheck.cli import app
from sitecheck.synthetic import SCENARIOS

ROOT = Path(__file__).resolve().parent.parent
README = (ROOT / "README.md").read_text()
CASE_STUDY_DATA = ROOT / "data" / "tcga_provgigapath.parquet"

failures: list[str] = []


def check(condition: bool, message: str) -> None:
    print(f"  {'ok  ' if condition else 'FAIL'} {message}")
    if not condition:
        failures.append(message)


print("verdicts")
code_verdicts = {expected for _, expected in SCENARIOS.values()}
documented = set(re.findall(r"`(CLEAN|SITE_SIGNAL_UNEXPLOITED|INFLATED|CONFOUNDED)`", README))
check(code_verdicts <= documented, f"every emitted verdict is documented ({code_verdicts})")
check(
    documented <= code_verdicts, f"no verdict documented that the code cannot emit ({documented})"
)

print("thresholds")
check(
    f"{SITE_ONLY_AUC_CONFOUNDED}" in README, f"site-only AUC threshold {SITE_ONLY_AUC_CONFOUNDED}"
)
check(f"{MIN_MATERIAL_DROP}" in README, f"material-drop threshold {MIN_MATERIAL_DROP}")
check(f"{MIN_SITE_N}" in README, f"minimum site size {MIN_SITE_N}")

print("cli invocations in the README parse")
click_cmd = get_command(app)
known = set(click_cmd.commands)
runner = CliRunner()
for block in re.findall(r"```bash\n(.*?)```", README, re.S):
    joined = re.sub(r"\\\n\s*", " ", block)
    for line in joined.splitlines():
        line = line.split("#")[0].strip()
        if not line.startswith("sitecheck "):
            continue
        argv = shlex.split(line)[1:]
        if not argv or argv[0] not in known:
            check(False, f"unknown subcommand: {line}")
            continue
        # --help proves the flags exist without needing the user's data file.
        res = runner.invoke(app, [argv[0], "--help"])
        unknown = [a for a in argv[1:] if a.startswith("--") and a.split("=")[0] not in res.stdout]
        check(not unknown, f"{argv[0]}: flags exist {argv[1:] and unknown or ''} <- {line}")

if CASE_STUDY_DATA.exists():
    print("tcga case-study table matches the code")
    out = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "tcga_case_study.py")],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    rows = [ln for ln in out.splitlines() if ln.startswith("| ") and "·" in ln]
    check(bool(rows), "case study produced rows")
    for row in rows:
        check(row.strip() in README, f"README row present: {row.split('|')[1].strip()}")
else:
    print(f"tcga case study skipped (no {CASE_STUDY_DATA.relative_to(ROOT)})")

print()
if failures:
    print(f"{len(failures)} README check(s) failed", file=sys.stderr)
    raise SystemExit(1)
print("README is consistent with the code")
