.PHONY: install fmt lint test demo doctor case-study readme clean

install:  ## uv sync with dev extras
	uv sync --extra dev

fmt:  ## ruff format
	uv run ruff format src tests scripts

lint:  ## ruff check + format check
	uv run ruff check src tests scripts
	uv run ruff format --check src tests scripts

test:  ## pytest (includes synthetic cohorts with known-correct verdicts)
	uv run pytest

doctor:  ## environment readiness check
	uv run sitecheck doctor --json

demo:  ## audit four synthetic cohorts; exits non-zero if a verdict regressed
	uv run sitecheck demo

case-study:  ## reproduce the README table on real TCGA data (downloads 445 MB)
	uv run python scripts/tcga_case_study.py

readme:  ## prove the README's TCGA table still matches the code
	uv run python scripts/test_readme.py

clean:
	rm -rf .pytest_cache .ruff_cache dist build **/__pycache__
