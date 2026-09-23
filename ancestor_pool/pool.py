"""The replacement-capable transaction pool.

Replacement rule (fixed subset, not a full RBF implementation)
--------------------------------------------------------------
When an incoming transaction spends an outpoint already spent by one or
more pool transactions *C*:

* the *removed set* is **C together with every descendant of C**
  (transitively);
* the incoming transaction is accepted only when

  1. it does not depend (explicit ``depends`` or spent creator outpoint)
     on any transaction in the removed set;
  2. ``fee(in) >  sum fee(removed)``           (strict);
  3. ``fee(in)/weight(in) >
     sum fee(removed)/sum weight(removed)``   (strict, exact cross
     multiplication -- no floats);

* all state changes are atomic: a rejected or failing ``add`` leaves the
  pool exactly as it was.

When no outpoint conflicts, the transaction is appended normally,
requiring every dependency to already exist in the pool.
"""

from .errors import (
    DuplicateTxIdError,
    MissingDependencyError,
    ReplacementRejectedError,
    ValidationError,
    REASON_FEE_NOT_GREATER,
    REASON_FEE_RATE_NOT_HIGHER,
    REASON_DEPENDS_ON_REMOVED,
)
from .graph import (
    build_edges,
    closure_maps,
    outpoint_creator,
    validate_incoming_batch,
)
from .model import validate_tx

REASON_REPLACEMENT_DISALLOWED = "replacement_disallowed"


class AddResult:
    """Outcome of a successful :meth:`AncestorPool.add`.

    Attributes:
        txid: id of the newly accepted transaction.
        removed_ids: frozenset of evicted ids (empty on a plain insert).
        replaced: ``True`` iff at least one transaction was evicted.
        removed_fee / removed_weight: aggregate totals of the removed set.
    """

    __slots__ = (
        "txid",
        "removed_ids",
        "replaced",
        "removed_fee",
        "removed_weight",
    )

    def __init__(self, txid, removed_ids, removed_fee, removed_weight):
        self.txid = txid
        self.removed_ids = frozenset(removed_ids)
        self.replaced = bool(self.removed_ids)
        self.removed_fee = removed_fee
        self.removed_weight = removed_weight

    def __repr__(self):
        return (
            f"AddResult(txid={self.txid!r}, replaced={self.replaced}, "
            f"removed_ids={sorted(self.removed_ids)!r})"
        )


class AncestorPool:
    """A bounded (max 30 transactions), replacement-aware in-memory pool."""

    MAX_TRANSACTIONS = 30

    def __init__(self):
        self._txs = {}
        self._parents = {}
        self._children = {}
        self._outpoint_owner = {}
        self._ancestors = {}
        self._descendants = {}

    # --- read-only views -------------------------------------------------

    def __len__(self):
        return len(self._txs)

    def __contains__(self, txid):
        return txid in self._txs

    def __iter__(self):
        return iter(self._txs)

    def get(self, txid):
        return self._txs.get(txid)

    def ids(self):
        return frozenset(self._txs)

    def parents(self, txid):
        return frozenset(self._parents.get(txid, ()))

    def children(self, txid):
        return frozenset(self._children.get(txid, ()))

    def ancestors(self, txid):
        return self._ancestors.get(txid, frozenset())

    def descendants(self, txid):
        return self._descendants.get(txid, frozenset())

    def spender_of(self, outpoint):
        """Return the txid currently spending ``outpoint`` or ``None``."""
        return self._outpoint_owner.get(outpoint)

    def edge_maps(self):
        """Return copies ``(txs, parents)`` for downstream algorithms."""
        return dict(self._txs), {k: set(v) for k, v in self._parents.items()}

    def _clone(self):
        clone = AncestorPool()
        clone._txs = dict(self._txs)
        clone._parents = {k: set(v) for k, v in self._parents.items()}
        clone._children = {k: set(v) for k, v in self._children.items()}
        clone._outpoint_owner = dict(self._outpoint_owner)
        clone._refresh_closures()
        return clone

    def _adopt(self, other):
        self._txs = other._txs
        self._parents = other._parents
        self._children = other._children
        self._outpoint_owner = other._outpoint_owner
        self._ancestors = other._ancestors
        self._descendants = other._descendants

    def _refresh_closures(self):
        self._ancestors, self._descendants = closure_maps(
            self._txs, self._parents
        )

    # --- mutation --------------------------------------------------------

    def add(self, tx, *, allow_replacement=True):
        """Add one transaction, applying the atomic replacement rule.

        Raises (state always left unchanged on error):
            ValidationError / DuplicateTxIdError / MissingDependencyError
            ReplacementRejectedError
        """
        validate_tx(tx, loc="tx")
        if tx.txid in self._txs:
            raise DuplicateTxIdError(tx.txid, loc="tx.txid")
        would_conflict = any(
            op in self._outpoint_owner for op in tx.inputs
        )
        # A replacement evicts at least one tx, so capacity cannot grow.
        if not would_conflict and len(self._txs) >= self.MAX_TRANSACTIONS:
            raise ValidationError(
                f"pool capacity {self.MAX_TRANSACTIONS} reached",
                loc="pool",
                kind="capacity",
            )
        return self._add(tx, allow_replacement=allow_replacement)

    def _add(self, tx, *, allow_replacement):
        # Snapshot for the atomicity guarantee.
        snap = (
            dict(self._txs),
            {k: set(v) for k, v in self._parents.items()},
            {k: set(v) for k, v in self._children.items()},
            dict(self._outpoint_owner),
        )
        try:
            result = self._commit_add(tx, allow_replacement)
        except Exception:
            (
                self._txs,
                self._parents,
                self._children,
                self._outpoint_owner,
            ) = snap
            self._refresh_closures()
            raise
        return result

    def _resolve_incoming_parents(self, tx):
        # Explicit dependencies must already live in the pool.
        deps = set(tx.depends)
        for op in tx.inputs:
            creator = outpoint_creator(op)
            if creator is None:
                continue  # bare outpoint name: external coin
            if creator == tx.txid:
                raise ValidationError(
                    f"transaction {tx.txid!r} must not spend its own "
                    f"output {op!r}",
                    "tx.inputs",
                    kind="cycle",
                )
            if creator in self._txs:
                deps.add(creator)
            # Unknown creator: external coin, no edge, no requirement.
        missing = sorted(d for d in deps if d not in self._txs)
        if missing:
            raise MissingDependencyError(
                f"transaction {tx.txid!r} references dependencies absent "
                f"from the pool: {missing}",
                missing=missing,
                loc="tx.depends",
            )
        return deps

    def _commit_add(self, tx, allow_replacement):
        incoming_parents = self._resolve_incoming_parents(tx)
        conflicts = {
            self._outpoint_owner[op]
            for op in tx.inputs
            if op in self._outpoint_owner
        }

        if conflicts and not allow_replacement:
            raise ReplacementRejectedError(
                tx.txid,
                self._removed_set(conflicts),
                REASON_REPLACEMENT_DISALLOWED,
                f"conflicts with {sorted(conflicts)} and replacement is "
                f"disabled",
            )

        if conflicts:
            removed = self._removed_set(conflicts)
            self._check_replacement_policy(tx, incoming_parents, removed)
            removed_fee = sum(self._txs[i].fee for i in removed)
            removed_weight = sum(self._txs[i].weight for i in removed)
            for rid in removed:
                del self._txs[rid]
                self._parents.pop(rid, None)
                self._children.pop(rid, None)
            for op, owner in list(self._outpoint_owner.items()):
                if owner in removed:
                    del self._outpoint_owner[op]
        else:
            removed = frozenset()
            removed_fee = 0
            removed_weight = 0

        self._txs[tx.txid] = tx
        parents, children = build_edges(self._txs)
        self._parents = parents
        self._children = children
        for op in tx.inputs:
            # No same-outpoint peer survives: direct conflicts were in the
            # removed set and descendants spent only created outpoints.
            if op in self._outpoint_owner:  # defensive, never expected
                raise RuntimeError(
                    f"internal error: outpoint {op!r} still contested "
                    f"after replacement"
                )
            self._outpoint_owner[op] = tx.txid
        self._refresh_closures()
        return AddResult(tx.txid, removed, removed_fee, removed_weight)

    def _removed_set(self, conflicts):
        removed = set(conflicts)
        for cid in conflicts:
            removed.update(self._descendants.get(cid, frozenset()))
        return frozenset(removed)

    def _check_replacement_policy(self, tx, incoming_parents, removed):
        # Structural check first, then absolute fee, then fee rate.
        on_removed = sorted(incoming_parents & removed)
        if on_removed:
            raise ReplacementRejectedError(
                tx.txid,
                removed,
                REASON_DEPENDS_ON_REMOVED,
                f"incoming tx depends on transactions that would be "
                f"evicted: {on_removed}",
            )
        removed_fee = sum(self._txs[i].fee for i in removed)
        removed_weight = sum(self._txs[i].weight for i in removed)
        if tx.fee <= removed_fee:
            raise ReplacementRejectedError(
                tx.txid,
                removed,
                REASON_FEE_NOT_GREATER,
                f"incoming fee {tx.fee} is not strictly greater than "
                f"removed-set fee {removed_fee} "
                f"(removed {sorted(removed)})",
            )
        # Exact rational comparison: f_in / w_in > f_rm / w_rm.
        if tx.fee * removed_weight <= removed_fee * tx.weight:
            raise ReplacementRejectedError(
                tx.txid,
                removed,
                REASON_FEE_RATE_NOT_HIGHER,
                f"incoming fee rate {tx.fee}/{tx.weight} is not strictly "
                f"greater than removed-set fee rate "
                f"{removed_fee}/{removed_weight} (removed "
                f"{sorted(removed)})",
            )

    def add_many(self, txs):
        """Atomic batch insert/replace.

        Every transaction is validated first; the batch is rejected
        wholesale (original pool untouched) if any element is invalid,
        ids repeat within the batch, two batch transactions spend the same
        outpoint, or any individual replacement rule fails when the
        transactions are applied in given order.
        """
        if isinstance(txs, (str, bytes)) or not hasattr(txs, "__iter__"):
            raise ValidationError(
                f"expected an iterable of Tx; got {type(txs).__name__}",
                "txs",
                kind="type_error",
            )
        txs = list(txs)

        # Full structural validation of the union of pool and batch
        # *before* any mutation; failure therefore leaves no partial
        # state behind.
        validate_incoming_batch(txs, self._txs, loc="txs")

        scratch = self._clone()
        results = []
        for tx in txs:
            results.append(scratch._add(tx, allow_replacement=True))
        self._adopt(scratch)
        return tuple(results)
