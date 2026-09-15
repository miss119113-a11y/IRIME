"""Public import shim for the paper implementation of SPFRIME.

Example
-------
    from spfrime import trap_dml_rime
"""
from __future__ import annotations

from pathlib import Path
import sys

_ROOT = Path(__file__).resolve().parent
_CORE = _ROOT / "experiments" / "final_core_execution"
_IMPL = _ROOT / "experiments" / "research" / "v7h_selective_subgrid_20260908"
for _p in (_CORE, _IMPL):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from v7h_selective_subgrid_rime import ConditionalDMLObjective, trap_dml_rime

__all__ = ["ConditionalDMLObjective", "trap_dml_rime"]
