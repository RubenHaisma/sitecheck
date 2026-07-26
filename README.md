# sitecheck

**Is your medical-ML result measuring biology, or the lab that produced the data?**

`sitecheck` answers that in one command. It probes how much of the submitting hospital
is readable from your features, how much of your label is explained by hospital
identity alone, and how far performance moves when the test hospital is unseen — then
gives a verdict you can gate CI on.

```bash
pip install sitecheck

sitecheck audit cohort.parquet \
  --features embedding --layer -1 \
  --label histological_grade --positive G3 \
  --tcga-barcode filename \
  --filter "cancer type abbreviation=STAD"
```

```
  460 rows · 393 patients · 22 sites · prevalence 60.9%

  site recoverable from features   0.814 balanced acc (chance 0.086) → leakage 0.80
  label explained by site alone    AUC 0.646 · Cramer's V 0.33
  random split (grouped)           AUC 0.765
  unseen site (6 held out)         AUC 0.644  [-0.122, 95% CI -0.172 to -0.071]

  verdict: INFLATED
```

## Why

Roughly **84%** of published oncology prognostic ML models are rated high risk of bias
([Dhiman 2022](https://pubmed.ncbi.nlm.nih.gov/35794668/)), only about **6%** are
externally validated ([Kim 2019](https://pubmed.ncbi.nlm.nih.gov/30799571/)), and of
those that are, **81% decline and 24% lose more than 0.10 AUC**
([Yu 2022](https://pubmed.ncbi.nlm.nih.gov/35652114/)).

A known driver is provenance. [Howard 2021](https://pubmed.ncbi.nlm.nih.gov/34285218/)
showed deep models identify the submitting TCGA site *after* stain normalisation, and
[PathoROB](https://pubmed.ncbi.nlm.nih.gov/42277006/) (Nature Communications, 2026)
found robustness deficits in **all 20** pathology foundation models it tested.

The diagnosis is established. What was missing is a tool that makes it cheap to check
your own cohort before you believe your own number. That's this.

## What it measures

| probe | question | why it matters |
|---|---|---|
| **site recoverability** | Can a linear probe read the hospital off the features? Run *within* each label class, so it can't cheat by proxying the outcome. | Non-zero means a stain/scanner fingerprint survives into your representation. |
| **label–site association** | Does hospital identity alone predict the label, scored out-of-fold? | If yes, the cohort cannot separate biology from provenance *at all* — no model fixes this. |
| **split sensitivity** | Random patient-grouped split vs. leave-one-hospital-out, with a paired-bootstrap CI on the difference. | This is the number that estimates what happens at a hospital you've never seen. |

### Verdicts

| verdict | meaning | what to do |
|---|---|---|
| `CLEAN` | Weak site fingerprint, performance holds on unseen hospitals. | Report your number. |
| `SITE_SIGNAL_UNEXPLOITED` | Fingerprint present, but this task isn't visibly using it. | Report both splits; note the fingerprint. |
| `INFLATED` | Performance drops materially at an unseen hospital. | Report the site-out number as your headline. |
| `CONFOUNDED` | Hospital alone predicts the label (site-only AUC ≥ 0.65). | The cohort can't answer this question. Change the cohort, not the model. |

A drop counts as material when the 95% CI excludes zero **and** the drop is at least
0.05 AUC — the band [Yu 2022](https://pubmed.ncbi.nlm.nih.gov/35652114/) used for a
meaningful external-validation decline. Significance alone isn't enough: a paired
bootstrap tightens with *n*, so on a large cohort a half-point drop would otherwise
trip the alarm.

## Real cohorts

Ten tasks on public TCGA Prov-GigaPath slide embeddings — 11,948 slides, 8,535
patients, 641 hospitals. One 445 MB parquet, no GPU, 28 seconds end to end
(`uv run python scripts/tcga_case_study.py`):

| task | n | sites | site-only AUC | leakage | random AUC | unseen-site AUC | delta (95% CI) | verdict |
|---|--:|--:|--:|--:|--:|--:|:--|---|
| KIRC · high grade | 899 | 6 | 0.640 | 0.68 | 0.645 | 0.601 | -0.044 (-0.077, -0.012) | SITE_SIGNAL_UNEXPLOITED |
| HNSC · high grade | 529 | 5 | 0.556 | 0.54 | 0.613 | 0.608 | -0.005 (-0.074, +0.064) | SITE_SIGNAL_UNEXPLOITED |
| LIHC · high grade | 430 | 4 | 0.617 | 0.57 | 0.761 | 0.659 | -0.102 (-0.159, -0.047) | INFLATED |
| STAD · high grade | 450 | 6 | 0.656 | 0.82 | 0.796 | 0.660 | -0.136 (-0.178, -0.097) | CONFOUNDED |
| KIRC · died within 2y | 777 | 6 | 0.657 | 0.73 | 0.718 | 0.681 | -0.037 (-0.070, -0.003) | CONFOUNDED |
| GBM · died within 2y | 826 | 7 | 0.530 | 0.69 | 0.570 | 0.454 | -0.116 (-0.167, -0.067) | INFLATED |
| LGG · died within 2y | 511 | 7 | 0.574 | 0.62 | 0.598 | 0.667 | +0.069 (-0.001, +0.138) | SITE_SIGNAL_UNEXPLOITED |
| HNSC · died within 2y | 424 | 3 | 0.538 | 0.48 | 0.550 | 0.438 | -0.113 (-0.195, -0.029) | INFLATED |
| UCEC · died within 2y | 411 | 6 | 0.661 | 0.82 | 0.571 | 0.465 | -0.106 (-0.216, +0.002) | CONFOUNDED |
| LUAD · died within 2y | 490 | 6 | 0.584 | 0.73 | 0.481 | 0.570 | +0.089 (+0.020, +0.156) | SITE_SIGNAL_UNEXPLOITED |

**6 of 10 flagged.** Three land on `CONFOUNDED`: knowing only which hospital submitted
the slide predicts the outcome nearly as well as a state-of-the-art foundation-model
embedding does. For UCEC 2-year survival it predicts it *better* — 0.661 from the
hospital label alone against 0.571 from the embedding.

Read the pattern honestly: these embeddings transfer well for *"what kind of tumour is
this"* and poorly for *"how will this patient do."* Four survival tasks land at or
below 0.5 on an unseen hospital.

## Python API

```python
from sitecheck import audit

report = audit(
    x,                  # (n, d) features
    y,                  # (n,)  binary labels
    site=hospitals,     # (n,)  hospital / scanner / batch
    patient=patients,   # (n,)  so one patient never straddles a split
)
print(report.verdict, report.reasons)
report.to_dict()        # JSON-ready
```

## CLI

Every command takes `--json` and emits exactly one JSON object; exit codes are
load-bearing (`0` ok, `1` error, `2` untrustworthy verdict under `--strict`).

```bash
sitecheck demo                  # four synthetic cohorts with known answers, no data needed
sitecheck audit DATA --features ... --label ... --site ... --patient ...
sitecheck audit DATA ... --strict   # exit 2 when the verdict is CONFOUNDED or INFLATED
sitecheck doctor --json         # environment check
```

`--tcga-barcode filename` derives hospital and patient from a TCGA barcode.
`--layer -1` picks one vector out of a per-layer embedding such as Prov-GigaPath's
`(14, 768)`. `--filter "col=value"` is repeatable.

## Scope and limits

- **Binary tasks only.** Multi-class and time-to-event are not implemented; don't infer
  a verdict for them from this.
- **The probe is a linear model on frozen features.** It measures what's *linearly*
  recoverable. A negative result bounds linear leakage, not all leakage.
- **`site` is whatever you pass.** Hospital, scanner, or batch — the audit is only as
  meaningful as that column.
- **A `CLEAN` verdict is not external validation.** It says this cohort's hospitals
  don't visibly drive this result. A genuinely independent cohort is still the standard.
- Leave-one-site-out needs sites with at least 25 rows and both classes present; sites
  below that are excluded and reported in `n_sites_evaluated`.

## Develop

```bash
make install    # uv sync --extra dev
make test       # pytest — includes four synthetic cohorts with known verdicts
make lint fmt
make demo       # sitecheck demo
make case-study # download 445 MB and audit ten real TCGA tasks
```

Apache-2.0.
