"""Synthetic cohorts where the correct verdict is known by construction.

Each builder plants one specific relationship between features, label and site. They
back both the test suite and ``sitecheck demo``, so the documented behaviour and the
tested behaviour cannot drift apart.
"""

from __future__ import annotations

import numpy as np

N_SITES = 8
PER_SITE = 70
DIM = 24
_SEED = 1234

Cohort = tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]


def _skeleton() -> tuple[np.random.Generator, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(_SEED)
    site = np.repeat([f"H{i}" for i in range(N_SITES)], PER_SITE)
    patient = np.array([f"p{i}" for i in range(len(site))])
    return rng, site, patient


def make_clean() -> Cohort:
    """Real label signal, no site fingerprint, label independent of site."""
    rng, site, patient = _skeleton()
    y = rng.integers(0, 2, size=len(site))
    direction = rng.normal(size=DIM)
    x = rng.normal(size=(len(site), DIM)) + 1.2 * y[:, None] * direction
    return x, y, site, patient


def make_site_signal_unexploited() -> Cohort:
    """Loud site fingerprint, orthogonal to a label signal that does transfer."""
    rng, site, patient = _skeleton()
    y = rng.integers(0, 2, size=len(site))
    basis = np.linalg.qr(rng.normal(size=(DIM, DIM)))[0]
    label_dir = basis[:, 0]
    site_dirs = {s: basis[:, 1 + i] for i, s in enumerate(np.unique(site))}
    offsets = np.stack([site_dirs[s] for s in site])
    x = rng.normal(scale=0.5, size=(len(site), DIM)) + 1.5 * y[:, None] * label_dir + 2.0 * offsets
    return x, y, site, patient


def make_confounded() -> Cohort:
    """Outcome rate is a property of the hospital, not of the tissue."""
    rng, site, patient = _skeleton()
    high = {f"H{i}" for i in range(N_SITES // 2)}
    prob = np.where(np.isin(site, list(high)), 0.9, 0.1)
    y = (rng.random(len(site)) < prob).astype(int)
    basis = np.linalg.qr(rng.normal(size=(DIM, DIM)))[0]
    offsets = np.stack([basis[:, 1 + i] for i in range(N_SITES) for _ in range(PER_SITE)])
    x = rng.normal(scale=0.5, size=(len(site), DIM)) + 5.0 * offsets
    return x, y, site, patient


def make_inflated() -> Cohort:
    """Each hospital encodes the label along its own axis, so nothing transfers.

    Label is balanced across hospitals, so site alone says nothing about outcome — the
    damage only shows up once the test hospital is unseen. This is the failure mode
    that a random split cannot see.
    """
    rng, site, patient = _skeleton()
    y = rng.integers(0, 2, size=len(site))
    basis = np.linalg.qr(rng.normal(size=(DIM, DIM)))[0]
    x = rng.normal(scale=0.5, size=(len(site), DIM))
    for i, s in enumerate(np.unique(site)):
        m = site == s
        x[m] += 5.0 * basis[:, i]  # hospital fingerprint
        x[m] += 3.0 * y[m][:, None] * basis[:, i + 1]  # hospital-specific label axis
    return x, y, site, patient


#: Builder → the verdict it must produce. Consumed by the tests and by ``demo``.
SCENARIOS: dict[str, tuple[callable, str]] = {
    "clean": (make_clean, "CLEAN"),
    "site_signal_unexploited": (make_site_signal_unexploited, "SITE_SIGNAL_UNEXPLOITED"),
    "inflated": (make_inflated, "INFLATED"),
    "confounded": (make_confounded, "CONFOUNDED"),
}
