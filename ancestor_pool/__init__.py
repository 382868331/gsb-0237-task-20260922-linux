"""Ancestor-package constrained mempool replacement (task-local subset).

Public surface:
    Tx, make_tx
    AncestorPool, AddResult
    select_package, SelectionResult, Exclusion
    select_optimal, OptimalSelectionResult, LedgerEntry
    MempoolError, ValidationError, DuplicateTxIdError,
    MissingDependencyError, ReplacementRejectedError

Only the small mathematical subset defined by this task is implemented;
it is not intended to be wire-compatible with any full node mempool.
"""

from .errors import (
    MempoolError,
    ValidationError,
    DuplicateTxIdError,
    MissingDependencyError,
    ReplacementRejectedError,
    REASON_FEE_NOT_GREATER,
    REASON_FEE_RATE_NOT_HIGHER,
    REASON_DEPENDS_ON_REMOVED,
)
from .model import Tx, make_tx, validate_tx
from .graph import validate_batch
from .pool import AncestorPool, AddResult
from .optimal import (
    select_optimal,
    OptimalSelectionResult,
    LedgerEntry,
    STATUS_OPTIMAL,
    MAX_CANDIDATES,
)
from .select import (
    select_package,
    SelectionResult,
    Exclusion,
    STATUS_PROVEN_OPTIMAL,
    STATUS_BUDGET_EXHAUSTED,
    CODE_OVERWEIGHT,
    CODE_ANCESTOR_EXCLUDED,
    CODE_NOT_SELECTED,
    CODE_BUDGET_INCONCLUSIVE,
    DEFAULT_BUDGET,
    MAX_TRANSACTIONS,
)

__all__ = [
    "Tx",
    "make_tx",
    "validate_tx",
    "validate_batch",
    "AncestorPool",
    "AddResult",
    "select_package",
    "select_optimal",
    "OptimalSelectionResult",
    "LedgerEntry",
    "STATUS_OPTIMAL",
    "MAX_CANDIDATES",
    "SelectionResult",
    "Exclusion",
    "MempoolError",
    "ValidationError",
    "DuplicateTxIdError",
    "MissingDependencyError",
    "ReplacementRejectedError",
    "REASON_FEE_NOT_GREATER",
    "REASON_FEE_RATE_NOT_HIGHER",
    "REASON_DEPENDS_ON_REMOVED",
    "STATUS_PROVEN_OPTIMAL",
    "STATUS_BUDGET_EXHAUSTED",
    "CODE_OVERWEIGHT",
    "CODE_ANCESTOR_EXCLUDED",
    "CODE_NOT_SELECTED",
    "CODE_BUDGET_INCONCLUSIVE",
    "DEFAULT_BUDGET",
    "MAX_TRANSACTIONS",
]
