"""Design data of the unit that both control and protection are configured with.

Kept apart from `control.py` so the protection limits can derive their thresholds from
the same ratings without importing the control loops.
"""

from __future__ import annotations

RATED_POWER_W: float = 300.0e6
RATED_STEAM_FLOW_KG_S: float = 245.0
