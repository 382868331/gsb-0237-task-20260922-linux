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
    ConfirmRejectedError,
    ValidationError,
    REASON_FEE_NOT_GREATER,
    REASON_FEE_RATE_NOT_HIGHER,
    REASON_DEPENDS_ON_REMOVED,
    CONFIRM_DUPLICATE_ID,
    CONFIRM_ALREADY_CONFIRMED,
    CONFIRM_DOUBLE_SPEND,
    CONFIRM_SPENDS_CONFIRMED,
    CONFIRM_BAD_ORDER,
)
from .graph import (
    build_edges,
    closure_maps,
    outpoint_creator,
    validate_incoming_batch,
)
from .model import validate_tx

REASON_REPLACEMENT_DISALLOWED = "replacement_disallowed"


class ConfirmResult:
    """Outcome of a successful :meth:`AncestorPool.confirm`.

    Attributes:
        confirmed_ids: txids accepted from the block, in given order.
        evicted_ids: pool txs removed because a block tx spends one of
            their inputs (each evicted tx plus all its descendants).
        retained_ids: pool txs that survive the confirmation.
        reparented_ids: surviving pool txs that had a pool parent which
            is now confirmed; they remain spending the confirmed output
            and the confirmed parent no longer counts in selection.
        created_outpoints: confirmed-tx-created outpoints referenced by
            a block/pool tx (the independently maintained set).
    """

    __slots__ = (
        "confirmed_ids",
        "evicted_ids",
        "retained_ids",
        "reparented_ids",
        "created_outpoints",
    )

    def __init__(
        self,
        confirmed_ids,
        evicted_ids,
        retained_ids,
        reparented_ids,
        created_outpoints,
    ):
        self.confirmed_ids = tuple(confirmed_ids)
        self.evicted_ids = frozenset(evicted_ids)
        self.retained_ids = frozenset(retained_ids)
        self.reparented_ids = frozenset(reparented_ids)
        self.created_outpoints = frozenset(created_outpoints)

    def __repr__(self):
        return (
            f"ConfirmResult(confirmed_ids={list(self.confirmed_ids)!r}, "
            f"evicted_ids={sorted(self.evicted_ids)!r}, "
            f"retained_ids={sorted(self.retained_ids)!r})"
        )


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
        # Ledger-confirmed state, maintained independently of the pool.
        self._confirmed_txs = {}
        self._confirmed_outpoints = set()
        self._confirmed_spent = set()

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

    def confirmed_ids(self):
        """Return the frozenset of ledger-confirmed txids."""
        return frozenset(self._confirmed_txs)

    def confirmed_outpoints(self):
        """Return a frozenset of known outpoints created by confirmed txs."""
        return frozenset(self._confirmed_outpoints)

    def is_outpoint_spent_confirmed(self, outpoint):
        """Whether a confirmed block transaction already spent ``outpoint``."""
        return outpoint in self._confirmed_spent

    def confirmed_parents(self, txid):
        """Confirmed txids whose outputs pool tx ``txid`` spends.

        Pool-internal :meth:`parents` lists only parents still in the
        pool; this exposes dependencies that have moved to the ledger.
        """
        tx = self._txs.get(txid)
        if tx is None:
            return frozenset()
        return frozenset(
            creator
            for op in tx.inputs
            if (creator := outpoint_creator(op)) in self._confirmed_txs
        )

    def edge_maps(self):
        """Return copies ``(txs, parents)`` for downstream algorithms."""
        return dict(self._txs), {k: set(v) for k, v in self._parents.items()}

    def _clone(self):
        clone = AncestorPool()
        clone._txs = dict(self._txs)
        clone._parents = {k: set(v) for k, v in self._parents.items()}
        clone._children = {k: set(v) for k, v in self._children.items()}
        clone._outpoint_owner = dict(self._outpoint_owner)
        clone._confirmed_txs = dict(self._confirmed_txs)
        clone._confirmed_outpoints = set(self._confirmed_outpoints)
        clone._confirmed_spent = set(self._confirmed_spent)
        clone._refresh_closures()
        return clone

    def _adopt(self, other):
        self._txs = other._txs
        self._parents = other._parents
        self._children = other._children
        self._outpoint_owner = other._outpoint_owner
        self._ancestors = other._ancestors
        self._descendants = other._descendants
        self._confirmed_txs = other._confirmed_txs
        self._confirmed_outpoints = other._confirmed_outpoints
        self._confirmed_spent = other._confirmed_spent

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
        confirmed_collision = sorted(
            op for op in tx.inputs if op in self._confirmed_spent
        )
        if confirmed_collision:
            raise ValidationError(
                f"transaction {tx.txid!r} spends outpoint(s) already "
                f"spent by confirmed block transactions: "
                f"{confirmed_collision}",
                "tx.inputs",
                kind="confirmed_double_spend",
            )
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

    # --- block confirmation ---------------------------------------------

    def confirm(self, block_transactions):
        """Apply one outer-ledger-confirmed block atomically.

        Every transaction is validated first; the whole block is
        rejected (pool byte-for-byte unchanged) on:

        * non-Tx / invalid fields;
        * duplicate txids inside the block, or a txid already confirmed;
        * two block txs spending the same outpoint, or a block tx
          spending an outpoint already spent by an earlier block;
        * dependency-order violation: a block tx naming/spending a
          parent that appears later in the block (cycles included).

        On success, same-id pool txs are removed but their non-conflicting
        descendants are **kept**, re-pointed at the confirmed outputs.
        Pool txs whose inputs the block spends (other-id conflicts) are
        removed together with all their descendants, as in ``add``.
        Confirmed txs take priority and are never reported twice.

        Returns :class:`ConfirmResult`.
        """
        if isinstance(block_transactions, (str, bytes)) or not hasattr(
            block_transactions, "__iter__"
        ):
            raise ValidationError(
                "expected an iterable of Tx; got "
                f"{type(block_transactions).__name__}",
                "block_transactions",
                kind="type_error",
            )
        block = list(block_transactions)

        # Phase 1: pure validation, no mutation.
        self._validate_block(block)

        # Phase 2: apply on a scratch copy; adopt only on success.
        scratch = self._clone()
        result = scratch._commit_confirm(block)
        self._adopt(scratch)
        return result

    def _validate_block(self, block):
        seen_ids = set()
        spent = {}
        block_ids = set()
        for i, tx in enumerate(block):
            loc = f"block_transactions[{i}]"
            validate_tx(tx, loc)
            if tx.txid in seen_ids:
                raise ConfirmRejectedError(
                    CONFIRM_DUPLICATE_ID,
                    f"txid {tx.txid!r} appears more than once in the "
                    f"confirmation list",
                    loc=f"{loc}.txid",
                )
            if tx.txid in self._confirmed_txs:
                raise ConfirmRejectedError(
                    CONFIRM_ALREADY_CONFIRMED,
                    f"txid {tx.txid!r} was already confirmed by an "
                    f"earlier block",
                    loc=f"{loc}.txid",
                )
            seen_ids.add(tx.txid)
            for op in tx.inputs:
                if op in spent:
                    raise ConfirmRejectedError(
                        CONFIRM_DOUBLE_SPEND,
                        f"outpoint {op!r} is spent by both "
                        f"{spent[op]!r} and {tx.txid!r} in the same "
                        f"confirmation list",
                        loc=f"{loc}.inputs",
                    )
                if op in self._confirmed_spent:
                    raise ConfirmRejectedError(
                        CONFIRM_SPENDS_CONFIRMED,
                        f"transaction {tx.txid!r} spends outpoint "
                        f"{op!r} already spent by a confirmed block",
                        loc=f"{loc}.inputs",
                    )
                spent[op] = tx.txid
            block_ids.add(tx.txid)

        # Dependency order: every parent referenced inside the block must
        # occur strictly earlier in the given list.
        for i, tx in enumerate(block):
            loc = f"block_transactions[{i}]"
            for dep in tx.depends:
                if dep in block_ids and dep not in {
                    b.txid for b in block[:i]
                }:
                    raise ConfirmRejectedError(
                        CONFIRM_BAD_ORDER,
                        f"transaction {tx.txid!r} depends on {dep!r}, "
                        f"which is not confirmed before it in the list",
                        loc=f"{loc}.depends",
                    )
            for op in tx.inputs:
                creator = outpoint_creator(op)
                if (
                    creator is not None
                    and creator in block_ids
                    and creator not in {b.txid for b in block[:i]}
                ):
                    raise ConfirmRejectedError(
                        CONFIRM_BAD_ORDER,
                        f"transaction {tx.txid!r} spends outpoint "
                        f"{op!r} whose creator {creator!r} is not "
                        f"confirmed before it in the list",
                        loc=f"{loc}.inputs",
                    )

    def _commit_confirm(self, block):
        block_ids = [tx.txid for tx in block]
        block_id_set = set(block_ids)

        # A block tx wins every outpoint it spends. A pool tx with a
        # *different* id spending one of those outpoints is a conflict
        # loser and is evicted with all its descendants. A pool tx with
        # the same id as a block tx is that tx being confirmed: it just
        # leaves the pool (its own spends are superseded by the confirmed
        # version) and descendants spending its outputs are retained.
        block_spent = set()
        for tx in block:
            block_spent.update(tx.inputs)
        direct_conflicts = set()
        for op in block_spent:
            owner = self._outpoint_owner.get(op)
            if owner is not None and owner not in block_id_set:
                direct_conflicts.add(owner)
        evicted = set(direct_conflicts)
        for cid in direct_conflicts:
            evicted.update(self._descendants.get(cid, frozenset()))
        # A descendant evicted via conflict even if it shares no id.
        # Same-id pool entries are simply removed (their children stay).
        same_id_in_pool = {
            tid for tid in block_ids if tid in self._txs and tid not in evicted
        }

        old_parents = {
            tid: set(ps) for tid, ps in self._parents.items()
        }
        leaving = evicted | same_id_in_pool
        retained = [tid for tid in self._txs if tid not in leaving]

        for tid in leaving:
            del self._txs[tid]
            self._parents.pop(tid, None)
            self._children.pop(tid, None)
        for op, owner in list(self._outpoint_owner.items()):
            if owner in leaving:
                del self._outpoint_owner[op]

        for tx in block:
            self._confirmed_txs[tx.txid] = tx
            for op in tx.inputs:
                self._confirmed_spent.add(op)

        # The confirmed-outpoint set is maintained independently of the
        # conflict machinery: record created outpoints that are
        # referenced by the block itself or by surviving pool txs.
        referenced = set()
        for tx in block:
            for op in tx.inputs:
                referenced.add(op)
        for tid in retained:
            for op in self._txs[tid].inputs:
                referenced.add(op)
        for op in referenced:
            creator = outpoint_creator(op)
            if creator in block_id_set or creator in self._confirmed_txs:
                self._confirmed_outpoints.add(op)

        # Rebuild edges over surviving pool txs only; links to confirmed
        # creators drop out of pool parents automatically.
        self._parents, self._children = build_edges(self._txs)
        self._refresh_closures()

        confirmed_id_set = set(self._confirmed_txs)
        reparented = set()
        for tid in retained:
            lost = old_parents.get(tid, set()) - self._parents.get(tid, set())
            if lost & confirmed_id_set:
                reparented.add(tid)

        return ConfirmResult(
            block_ids,
            evicted,
            retained,
            reparented,
            set(self._confirmed_outpoints),
        )

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
