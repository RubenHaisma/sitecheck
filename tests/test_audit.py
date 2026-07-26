"""Verdicts on synthetic cohorts whose right answer is known by construction.

The cohort builders live in ``sitecheck.synthetic`` so that ``sitecheck demo`` and this
suite exercise exactly the same fixtures.
"""

from __future__ import annotations

import numpy as np
import pytest

from sitecheck import audit
from sitecheck.synthetic import (
    N_SITES,
    PER_SITE,
    make_clean,
    make_confounded,
    make_inflated,
    make_site_signal_unexploited,
)


def _rng() -> np.random.Generator:
    return np.random.default_rng(1234)


def _skeleton() -> tuple[np.ndarray, np.ndarray]:
    site = np.repeat([f"H{i}" for i in range(N_SITES)], PER_SITE)
    patient = np.array([f"p{i}" for i in range(len(site))])
    return site, patient


# --------------------------------------------------------------------- verdicts


def test_clean_cohort_passes():
    rep = audit(*make_clean())
    assert rep.verdict == "CLEAN", rep.reasons
    assert rep.site_recoverability.leakage_score < 0.30
    assert not rep.label_site_association.is_confounded


def test_site_fingerprint_without_exploitation():
    rep = audit(*make_site_signal_unexploited())
    assert rep.verdict == "SITE_SIGNAL_UNEXPLOITED", rep.reasons
    # The fingerprint is loud...
    assert rep.site_recoverability.leakage_score > 0.60
    # ...but the label signal still transfers to an unseen hospital.
    assert rep.split_sensitivity.site_out_auc > 0.80


def test_confounded_cohort_is_caught_before_anything_else():
    rep = audit(*make_confounded())
    assert rep.verdict == "CONFOUNDED", rep.reasons
    assert rep.label_site_association.site_only_auc > 0.65
    assert rep.label_site_association.cramers_v > 0.5


def test_inflated_cohort_is_flagged():
    rep = audit(*make_inflated())
    assert rep.verdict == "INFLATED", rep.reasons
    assert rep.split_sensitivity.delta < 0
    assert rep.split_sensitivity.is_material_drop
    # Site alone must NOT explain the label here — that is what separates this
    # scenario from CONFOUNDED.
    assert rep.label_site_association.site_only_auc < 0.65


# ------------------------------------------------------------------ properties


def test_within_label_probe_ignores_a_label_shaped_feature():
    """A feature that encodes only the label must not register as site leakage."""
    rng = _rng()
    site, patient = _skeleton()
    y = rng.integers(0, 2, size=len(site))
    x = np.column_stack([y * 10.0, rng.normal(size=(len(site), 4))])
    rep = audit(x, y, site, patient)
    assert rep.site_recoverability.leakage_score < 0.30


def test_patient_grouping_changes_the_random_split():
    """Duplicating each patient inflates a naive split; passing `patient` prevents it."""
    x, y, site, _ = make_clean()
    jitter = np.random.default_rng(7).normal(scale=0.01, size=(2 * len(x), x.shape[1]))
    x2 = np.repeat(x, 2, axis=0) + jitter
    y2, site2 = np.repeat(y, 2), np.repeat(site, 2)
    grouped = np.repeat([f"p{i}" for i in range(len(x))], 2)
    ungrouped = np.array([f"p{i}" for i in range(len(x2))])
    leaky = audit(x2, y2, site2, ungrouped).split_sensitivity.random_auc
    honest = audit(x2, y2, site2, grouped).split_sensitivity.random_auc
    assert leaky >= honest


def test_report_round_trips_to_json():
    import json

    rep = audit(*make_clean())
    payload = json.loads(json.dumps(rep.to_dict()))
    assert payload["verdict"] == "CLEAN"
    assert set(payload) >= {"n", "n_sites", "site_recoverability", "split_sensitivity"}


# ------------------------------------------------------------------ validation


def test_rejects_non_binary_labels():
    x, _, site, patient = make_clean()
    y = np.random.default_rng(0).integers(0, 3, size=len(site))
    with pytest.raises(ValueError, match="binary"):
        audit(x, y, site, patient)


def test_rejects_length_mismatch():
    x, y, site, patient = make_clean()
    with pytest.raises(ValueError, match="length mismatch"):
        audit(x, y[:-1], site, patient)


def test_rejects_1d_features():
    _, y, site, patient = make_clean()
    with pytest.raises(ValueError, match="2-D"):
        audit(np.zeros(len(y)), y, site, patient)


def test_patient_defaults_to_one_per_row():
    x, y, site, _ = make_clean()
    rep = audit(x, y, site)
    assert rep.n_patients == rep.n
