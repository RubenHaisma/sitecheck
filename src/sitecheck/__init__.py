"""sitecheck — audit whether a medical-ML result measures biology or the lab.

from sitecheck import audit
print(audit(x, y, site=hospital_ids, patient=patient_ids))
"""

from sitecheck.audit import (
    AuditReport,
    LabelSiteAssociation,
    SiteRecoverability,
    SplitSensitivity,
    audit,
)

__all__ = [
    "audit",
    "AuditReport",
    "SiteRecoverability",
    "LabelSiteAssociation",
    "SplitSensitivity",
]
__version__ = "0.1.0"
