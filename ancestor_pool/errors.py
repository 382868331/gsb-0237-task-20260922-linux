"""Error hierarchy and stable reason codes.

All errors carry a ``loc`` path (e.g. ``"txs[3].fee"`` or
``"block_weight"``) so that invalid input is locatable rather than
silently corrected.
"""


# --- replacement failure reason codes (stable, public strings) ---
REASON_FEE_NOT_GREATER = "fee_not_strictly_greater"
REASON_FEE_RATE_NOT_HIGHER = "fee_rate_not_strictly_higher"
REASON_DEPENDS_ON_REMOVED = "depends_on_removed_set"

# --- confirm-block failure reason codes (stable, public strings) ---
CONFIRM_DUPLICATE_ID = "duplicate_id_in_block"
CONFIRM_ALREADY_CONFIRMED = "txid_already_confirmed"
CONFIRM_DOUBLE_SPEND = "block_double_spend"
CONFIRM_SPENDS_CONFIRMED = "spends_already_confirmed_outpoint"
CONFIRM_BAD_ORDER = "dependency_order_violation"


class MempoolError(Exception):
    """Base class for all errors raised by this package."""


class ValidationError(MempoolError, ValueError):
    """A caller-supplied value violates the public data contract.

    Attributes:
        loc: dotted/indexed path to the offending field, or ``"?"``.
        kind: short stable machine-readable category.
    """

    def __init__(self, message, loc="?", kind="invalid_value"):
        self.loc = loc
        self.kind = kind
        full = f"[{loc}] {message}" if loc not in (None, "?") else message
        super().__init__(full)


class DuplicateTxIdError(MempoolError):
    """Raised when a batch contains the same tx id twice."""

    def __init__(self, txid, loc="?"):
        self.txid = txid
        self.loc = loc
        super().__init__(f"[{loc}] duplicate transaction id: {txid!r}")


class MissingDependencyError(MempoolError):
    """A tx depends on an id / outpoint that is absent from the batch/pool."""

    def __init__(self, message, missing=(), loc="?"):
        self.missing = tuple(missing)
        self.loc = loc
        super().__init__(f"[{loc}] {message}" if loc not in (None, "?") else message)


class ReplacementRejectedError(MempoolError):
    """Raised when an ``add`` replacement is refused (pool left unchanged).

    Attributes:
        incoming_id: id of the rejected transaction.
        removed_ids: conflicting set that *would* have been evicted
            (conflicting txs and all their descendants).
        reason: one of the ``REASON_*`` stable codes.
        detail: human-readable explanation with the compared numbers.
    """

    def __init__(self, incoming_id, removed_ids, reason, detail):
        self.incoming_id = incoming_id
        self.removed_ids = frozenset(removed_ids)
        self.reason = reason
        self.detail = detail
        super().__init__(
            f"replacement by {incoming_id!r} rejected ({reason}): {detail}"
        )


class ConfirmRejectedError(MempoolError):
    """Raised when a ``confirm`` batch is invalid (pool left unchanged).

    Attributes:
        reason: one of the ``CONFIRM_*`` stable codes.
        loc: dotted/indexed path to the offending entry, or ``"?"``.
    """

    def __init__(self, reason, detail, loc="?"):
        self.reason = reason
        self.loc = loc
        full = f"[{loc}] " if loc not in (None, "?") else ""
        super().__init__(
            f"{full}block confirmation rejected ({reason}): {detail}"
        )
