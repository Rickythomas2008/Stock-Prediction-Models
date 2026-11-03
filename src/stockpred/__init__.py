"""A-share daily-frequency prediction and decision support.

This package is the engineering spine that the legacy notebooks in this
repository lack: data acquisition with validation, freshness guards, run
records and (in later slices) evaluation, risk gates and execution.

Nothing in this package places orders on its own. Execution requires an
explicit broker confirmation state (see ``stockpred.errors.BrokerNotConfirmed``).
"""

from __future__ import annotations

__all__ = ["__version__", "SOFTWARE_NAME", "DEVELOPER"]

SOFTWARE_NAME = "stockpred"

__version__ = "0.1.0"

#: Registered as "开发主体" in the program-trading filing (CSRC 2024 No.8, art.7).
DEVELOPER = "Rickythomas2008"
