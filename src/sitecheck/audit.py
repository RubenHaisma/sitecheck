"""Core audit: is a medical-ML result measuring biology, or the lab that made the data?

Three probes, then a verdict:

1. ``site_recoverability`` — can a linear probe read the submitting site off the
   features? Run *within* each label class, so it cannot cheat by proxying the label.
2. ``label_site_association`` — does site alone predict the label? If it does, no
   model on this cohort can separate biology from provenance. This is the probe that
   decides whether the question is answerable at all.
3. ``split_sensitivity`` — how much does performance move when the test site is
   unseen, with a paired-bootstrap CI on the difference.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler

# A site needs enough slides for a held-out fold to mean anything, and both classes
# present or its AUC is undefined.
MIN_SITE_N = 25
# The within-label site probe needs a few examples per site to fit a fold at all.
MIN_SITE_N_PER_LABEL = 3
# Above this, knowing only the hospital tells you the label — the cohort cannot
# answer the biological question regardless of model.
SITE_ONLY_AUC_CONFOUNDED = 0.65
# Normalised site-recoverability above this counts as "the features carry the lab".
LEAKAGE_SCORE_HIGH = 0.30
# A paired bootstrap CI tightens with n, so significance alone would escalate a
# half-point AUC drop on a large cohort. Require a materially large drop too. 0.05 is
# the band Yu et al. 2022 (PMID 35652114) used for a meaningful external-validation
# decline, so a verdict here means the same thing it means in that literature.
MIN_MATERIAL_DROP = 0.05
N_BOOTSTRAP = 2000
_SEED = 0


@dataclass
class SiteRecoverability:
    """How much of the submitting site is readable from the features."""

    balanced_accuracy: float
    chance: float
    leakage_score: float
    n_sites: int
    within_label: bool

    @property
    def is_high(self) -> bool:
        return self.leakage_score >= LEAKAGE_SCORE_HIGH


@dataclass
class LabelSiteAssociation:
    """How much of the label is explained by site membership alone."""

    site_only_auc: float
    cramers_v: float

    @property
    def is_confounded(self) -> bool:
        return self.site_only_auc >= SITE_ONLY_AUC_CONFOUNDED


@dataclass
class SplitSensitivity:
    """Performance under a random split vs. an unseen-site split."""

    random_auc: float
    site_out_auc: float
    site_out_auc_macro: float
    delta: float
    delta_ci95: tuple[float, float]
    n_evaluated: int
    n_sites_evaluated: int
    per_site: dict[str, float] = field(default_factory=dict)

    @property
    def is_material_drop(self) -> bool:
        """True when the drop is both statistically clear and large enough to matter."""
        if np.isnan(self.delta) or np.isnan(self.delta_ci95[1]):
            return False
        return self.delta_ci95[1] < 0.0 and self.delta <= -MIN_MATERIAL_DROP


@dataclass
class AuditReport:
    n: int
    n_patients: int
    n_sites: int
    label_prevalence: float
    site_recoverability: SiteRecoverability
    label_site_association: LabelSiteAssociation
    split_sensitivity: SplitSensitivity
    verdict: str
    reasons: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _fit_probe(x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray, *, proba: bool):
    scaler = StandardScaler().fit(x_train)
    clf = LogisticRegression(max_iter=2000).fit(scaler.transform(x_train), y_train)
    xt = scaler.transform(x_test)
    return clf.predict_proba(xt)[:, 1] if proba else clf.predict(xt)


def _site_recoverability(
    x: np.ndarray, y: np.ndarray, site: np.ndarray, patient: np.ndarray
) -> SiteRecoverability:
    """Predict site from features within each label class, then pool the scores.

    Holding the label fixed is what makes this a site probe rather than a relabelled
    version of the outcome task.
    """
    scores: list[tuple[float, float, int]] = []
    for label in np.unique(y):
        m = y == label
        xs, ss, ps = x[m], site[m], patient[m]
        site_names, site_counts = np.unique(ss, return_counts=True)
        keep = np.isin(ss, site_names[site_counts >= MIN_SITE_N_PER_LABEL])
        xs, ss, ps = xs[keep], ss[keep], ps[keep]
        if len(np.unique(ss)) < 2 or len(ss) < 10:
            continue
        n_splits = min(5, int(np.min(np.unique(ss, return_counts=True)[1])), len(np.unique(ps)))
        if n_splits < 2:
            continue
        cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=_SEED)
        pred = np.empty(len(ss), dtype=object)
        try:
            for tr, te in cv.split(xs, ss, groups=ps):
                pred[te] = _fit_probe(xs[tr], ss[tr], xs[te], proba=False)
        except ValueError:
            continue
        scores.append((balanced_accuracy_score(ss, pred), 1.0 / len(np.unique(ss)), len(ss)))

    if not scores:
        return SiteRecoverability(float("nan"), float("nan"), float("nan"), 0, True)

    w = np.array([s[2] for s in scores], dtype=float)
    acc = float(np.average([s[0] for s in scores], weights=w))
    chance = float(np.average([s[1] for s in scores], weights=w))
    # Normalise so 0 = chance and 1 = perfect, making the number comparable across
    # cohorts with different site counts.
    leakage = max(0.0, (acc - chance) / (1.0 - chance)) if chance < 1.0 else 0.0
    return SiteRecoverability(acc, chance, leakage, len(np.unique(site)), True)


def _label_site_association(
    y: np.ndarray, site: np.ndarray, patient: np.ndarray
) -> LabelSiteAssociation:
    """Predict the label from site identity alone, out-of-fold.

    Out-of-fold target encoding, not in-sample site means: a site's own outcome rate
    would trivially predict its own members.
    """
    n_splits = min(5, len(np.unique(patient)))
    p = np.full(len(y), y.mean(), dtype=float)
    if n_splits >= 2:
        cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=_SEED)
        for tr, te in cv.split(np.zeros((len(y), 1)), y, groups=patient):
            rates = {s: y[tr][site[tr] == s].mean() for s in np.unique(site[tr])}
            gm = float(y[tr].mean())
            p[te] = [rates.get(s, gm) for s in site[te]]
    auc = float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else float("nan")

    # Cramér's V between site and label, bias-corrected.
    sites, labels = np.unique(site), np.unique(y)
    table = np.array([[np.sum((site == s) & (y == lab)) for lab in labels] for s in sites])
    n = table.sum()
    if n == 0 or table.shape[0] < 2 or table.shape[1] < 2:
        return LabelSiteAssociation(auc, float("nan"))
    expected = np.outer(table.sum(1), table.sum(0)) / n
    chi2 = float(np.sum((table - expected) ** 2 / np.where(expected == 0, np.nan, expected)))
    chi2 = 0.0 if np.isnan(chi2) else chi2
    phi2 = chi2 / n
    r, k = table.shape
    phi2c = max(0.0, phi2 - (k - 1) * (r - 1) / (n - 1))
    rc, kc = r - (r - 1) ** 2 / (n - 1), k - (k - 1) ** 2 / (n - 1)
    denom = min(kc - 1, rc - 1)
    v = float(np.sqrt(phi2c / denom)) if denom > 0 else float("nan")
    return LabelSiteAssociation(auc, v)


def _paired_bootstrap_delta(
    y: np.ndarray, p_a: np.ndarray, p_b: np.ndarray, rng: np.random.Generator
) -> tuple[float, float]:
    """95% CI for AUC(b) - AUC(a), resampling the same rows for both predictors."""
    deltas = []
    idx = np.arange(len(y))
    for _ in range(N_BOOTSTRAP):
        b = rng.choice(idx, size=len(idx), replace=True)
        if len(np.unique(y[b])) < 2:
            continue
        deltas.append(roc_auc_score(y[b], p_b[b]) - roc_auc_score(y[b], p_a[b]))
    if not deltas:
        return (float("nan"), float("nan"))
    lo, hi = np.percentile(deltas, [2.5, 97.5])
    return (float(lo), float(hi))


def _split_sensitivity(
    x: np.ndarray, y: np.ndarray, site: np.ndarray, patient: np.ndarray
) -> SplitSensitivity:
    names, counts_arr = np.unique(site, return_counts=True)
    counts = dict(zip(names, counts_arr, strict=True))
    usable = [
        s
        for s, c in counts.items()
        if c >= MIN_SITE_N and len(np.unique(y[site == s])) > 1 and len(np.unique(y[site != s])) > 1
    ]

    # Random split, grouped by patient so one patient never straddles the boundary.
    n_splits = min(5, len(np.unique(patient)), int(np.min(np.unique(y, return_counts=True)[1])))
    p_random = np.full(len(y), np.nan)
    if n_splits >= 2:
        cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=_SEED)
        for tr, te in cv.split(x, y, groups=patient):
            p_random[te] = _fit_probe(x[tr], y[tr], x[te], proba=True)

    # Leave-one-site-out.
    p_site = np.full(len(y), np.nan)
    per_site: dict[str, float] = {}
    for s in usable:
        te = site == s
        p_site[te] = _fit_probe(x[~te], y[~te], x[te], proba=True)
        per_site[str(s)] = float(roc_auc_score(y[te], p_site[te]))

    both = ~np.isnan(p_random) & ~np.isnan(p_site)
    if both.sum() < 10 or len(np.unique(y[both])) < 2:
        nan = float("nan")
        return SplitSensitivity(
            nan, nan, nan, nan, (nan, nan), int(both.sum()), len(usable), per_site
        )

    a_random = float(roc_auc_score(y[both], p_random[both]))
    a_site = float(roc_auc_score(y[both], p_site[both]))
    macro = float(np.mean(list(per_site.values()))) if per_site else float("nan")
    ci = _paired_bootstrap_delta(
        y[both], p_random[both], p_site[both], np.random.default_rng(_SEED)
    )
    return SplitSensitivity(
        a_random, a_site, macro, a_site - a_random, ci, int(both.sum()), len(usable), per_site
    )


def _decide(
    rec: SiteRecoverability, assoc: LabelSiteAssociation, split: SplitSensitivity
) -> tuple[str, list[str]]:
    reasons: list[str] = []

    if assoc.is_confounded:
        reasons.append(
            f"site alone predicts the label at AUC {assoc.site_only_auc:.3f} "
            f"(>= {SITE_ONLY_AUC_CONFOUNDED}) — provenance and outcome are entangled in this "
            f"cohort, so no model on it can attribute performance to biology"
        )
        return "CONFOUNDED", reasons

    if rec.is_high:
        reasons.append(
            f"a linear probe recovers the submitting site from the features at "
            f"{rec.balanced_accuracy:.3f} balanced accuracy vs {rec.chance:.3f} chance "
            f"(leakage score {rec.leakage_score:.2f}), holding the label fixed"
        )
        if split.is_material_drop:
            reasons.append(
                f"and AUC falls {abs(split.delta):.3f} at an unseen site "
                f"(95% CI [{split.delta_ci95[0]:.3f}, {split.delta_ci95[1]:.3f}]) — the "
                f"random-split figure is optimistic for a new hospital"
            )
            return "INFLATED", reasons
        reasons.append(
            f"but the change at an unseen site is not material "
            f"({split.delta:+.3f}, 95% CI [{split.delta_ci95[0]:.3f}, "
            f"{split.delta_ci95[1]:.3f}]) — site signal is present but this task is not "
            f"visibly exploiting it"
        )
        return "SITE_SIGNAL_UNEXPLOITED", reasons

    if split.is_material_drop:
        reasons.append(
            f"site is not strongly recoverable (leakage score {rec.leakage_score:.2f}) yet AUC "
            f"still falls {abs(split.delta):.3f} at an unseen site — suspect distribution shift "
            f"in the label definition or case mix rather than a stain/scanner signature"
        )
        return "INFLATED", reasons

    reasons.append(
        f"site is weakly recoverable (leakage score {rec.leakage_score:.2f}) and performance "
        f"holds at an unseen site ({split.delta:+.3f}, 95% CI "
        f"[{split.delta_ci95[0]:.3f}, {split.delta_ci95[1]:.3f}])"
    )
    return "CLEAN", reasons


def audit(
    x: np.ndarray,
    y: np.ndarray,
    site: np.ndarray,
    patient: np.ndarray | None = None,
) -> AuditReport:
    """Audit a binary medical-ML task for site confounding.

    Args:
        x: Feature matrix ``(n, d)`` — embeddings, radiomics, whatever the model consumes.
        y: Binary labels ``(n,)``.
        site: Provenance of each row ``(n,)`` — hospital, scanner, or batch.
        patient: Patient identifier ``(n,)``. Defaults to one patient per row; pass it
            whenever several rows can come from the same person, or the random split
            leaks that person across the boundary.

    Returns:
        An :class:`AuditReport` whose ``verdict`` is one of ``CONFOUNDED``,
        ``INFLATED``, ``SITE_SIGNAL_UNEXPLOITED`` or ``CLEAN``.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y)
    site = np.asarray(site).astype(str)
    patient = np.arange(len(y)).astype(str) if patient is None else np.asarray(patient).astype(str)

    if x.ndim != 2:
        raise ValueError(f"x must be 2-D (n, d); got shape {x.shape}")
    if not (len(x) == len(y) == len(site) == len(patient)):
        raise ValueError(
            f"length mismatch: x={len(x)} y={len(y)} site={len(site)} patient={len(patient)}"
        )
    classes = np.unique(y)
    if len(classes) != 2:
        raise ValueError(f"audit() handles binary labels; got {len(classes)} classes: {classes}")

    y = (y == classes[1]).astype(int)

    rec = _site_recoverability(x, y, site, patient)
    assoc = _label_site_association(y, site, patient)
    split = _split_sensitivity(x, y, site, patient)
    verdict, reasons = _decide(rec, assoc, split)

    return AuditReport(
        n=len(y),
        n_patients=int(len(np.unique(patient))),
        n_sites=int(len(np.unique(site))),
        label_prevalence=float(y.mean()),
        site_recoverability=rec,
        label_site_association=assoc,
        split_sensitivity=split,
        verdict=verdict,
        reasons=reasons,
    )
