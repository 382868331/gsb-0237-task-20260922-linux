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


def _validate_outpoints(value, loc):
    if not isinstance(value, (tuple, list)):
        raise ValidationError(
            f"created outpoints must be a tuple or list of str; got "
            f"{type(value).__name__}",
            loc,
            kind="type_error",
        )
    seen = set()
    for i, op in enumerate(value):
        iloc = f"{loc}[{i}]"
        if not isinstance(op, str) or op == "" or "\x00" in op:
            raise ValidationError(
                f"created outpoint must be a non-empty str without NUL; "
                f"got {op!r}",
                iloc,
                kind="type_error",
            )
        if op in seen:
            raise ValidationError(
                f"duplicate created outpoint {op!r}",
                iloc,
                kind="duplicate",
            )
        seen.add(op)
    return tuple(value)


class BlockTx:
    """One confirmed transaction and the outpoints it creates.

    The outer ledger confirms transactions; the pool only needs the
    transaction itself (for id, consumed outpoints and dependencies)
    plus the explicit list of outpoints it creates, since Tx does not
    declare outputs.
    """

    __slots__ = ("tx", "created")

    def __init__(self, tx, created=()):
        self.tx = tx
        self.created = tuple(created)

    def __repr__(self):
        return f"BlockTx(tx={self.tx!r}, created={self.created!r})"


class ConfirmResult:
    """Outcome of a successful AncestorPool.confirm.

    Attributes:
        confirmed_ids: all block transaction ids, in block order.
        pool_confirmed_ids: block ids that were present in the pool and
            were removed *by confirmation* (their pool descendants are
            kept and now depend on confirmed outputs).
        conflict_removed_ids: pool ids evicted because they (or an
            ancestor) double-spent a block transaction's input. Ids
            confirmed in the same block are never reported here.
    """

    __slots__ = ("confirmed_ids", "pool_confirmed_ids", "conflict_removed_ids")

    def __init__(self, confirmed_ids, pool_confirmed_ids, conflict_removed_ids):
        self.confirmed_ids = tuple(confirmed_ids)
        self.pool_confirmed_ids = frozenset(pool_confirmed_ids)
        self.conflict_removed_ids = frozenset(conflict_removed_ids)

    @property
    def removed_ids(self):
        """Every pool id evicted by this confirmation."""
        return self.pool_confirmed_ids | self.conflict_removed_ids

    def __repr__(self):
        return (
            f"ConfirmResult(confirmed_ids={list(self.confirmed_ids)!r}, "
            f"pool_confirmed_ids={sorted(self.pool_confirmed_ids)!r}, "
            f"conflict_removed_ids={sorted(self.conflict_removed_ids)!r})"
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
        # Confirmed-ledger view (deliberately minimal, not a full ledger):
        # ids ever confirmed, outpoints created and still unspent, and
        # outpoints consumed by confirmed transactions.
        self._confirmed_ids = set()
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

    def is_confirmed(self, txid):
        """Return True iff txid was confirmed by an outer block."""
        return txid in self._confirmed_ids

    def confirmed_outpoints(self):
        """Frozenset of confirmed-created, still-unspent outpoints."""
        return frozenset(self._confirmed_outpoints)

    def confirmed_spent(self):
        """Frozenset of outpoints consumed by confirmed transactions."""
        return frozenset(self._confirmed_spent)

    def edge_maps(self):
        """Return copies ``(txs, parents)`` for downstream algorithms."""
        return dict(self._txs), {k: set(v) for k, v in self._parents.items()}

    def _clone(self):
        clone = AncestorPool()
        clone._txs = dict(self._txs)
        clone._parents = {k: set(v) for k, v in self._parents.items()}
        clone._children = {k: set(v) for k, v in self._children.items()}
        clone._outpoint_owner = dict(self._outpoint_owner)
        clone._confirmed_ids = set(self._confirmed_ids)
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
        self._confirmed_ids = other._confirmed_ids
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
        if tx.txid in self._confirmed_ids:
            raise DuplicateTxIdError(tx.txid, loc="tx.txid")
        for op in tx.inputs:
            if op in self._confirmed_spent:
                raise ValidationError(
                    f"outpoint {op!r} was already spent by a confirmed "
                    f"transaction",
                    "tx.inputs",
                    kind="double_spend",
                )
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

    # --- confirmation ----------------------------------------------------

    def confirm(self, block):
        """Apply a confirmed block atomically.

        Args:
            block: iterable of :class:`BlockTx` (or `(tx, created)`
                pairs) in ledger order. Each item carries the confirmed
                transaction and the outpoints it creates.

        Effects (all-or-nothing):
            * pool transactions whose id is confirmed are removed; their
              pool descendants stay and now depend on confirmed outputs;
            * pool transactions double-spending a block input are
              removed together with all their descendants, unless they
              are themselves confirmed in this block (confirmation wins
              and is reported only once);
            * the confirmed id/outpoint sets are updated.

        The whole block is validated first (duplicate ids, duplicate
        created outpoints, intra-block double spends, spends of
        previously confirmed-spent outpoints, dependency order); any
        violation raises and leaves the pool untouched.
        """
        items = self._normalize_block(block)
        self._validate_block(items)
        scratch = self._clone()
        result = scratch._commit_confirm(items)
        self._adopt(scratch)
        return result

    @staticmethod
    def _normalize_block(block):
        if isinstance(block, (str, bytes)) or not hasattr(block, "__iter__"):
            raise ValidationError(
                f"expected an iterable of BlockTx; got "
                f"{type(block).__name__}",
                "block",
                kind="type_error",
            )
        items = []
        for i, entry in enumerate(block):
            loc = f"block[{i}]"
            if isinstance(entry, BlockTx):
                btx = entry
            elif isinstance(entry, (tuple, list)) and len(entry) == 2:
                btx = BlockTx(entry[0], entry[1])
            else:
                raise ValidationError(
                    f"block entries must be BlockTx or (tx, created) "
                    f"pairs; got {type(entry).__name__}",
                    loc,
                    kind="type_error",
                )
            validate_tx(btx.tx, loc=f"{loc}.tx")
            created = _validate_outpoints(btx.created, f"{loc}.created")
            items.append(BlockTx(btx.tx, created))
        return items

    def _validate_block(self, items):
        seen_ids = set()
        seen_created = set()
        spent_in_block = set()
        for i, btx in enumerate(items):
            tx = btx.tx
            loc = f"block[{i}]"
            if tx.txid in seen_ids:
                raise DuplicateTxIdError(tx.txid, loc=f"{loc}.tx.txid")
            if tx.txid in self._confirmed_ids:
                raise DuplicateTxIdError(tx.txid, loc=f"{loc}.tx.txid")
            seen_ids.add(tx.txid)
            for op in btx.created:
                if op in seen_created or op in self._confirmed_outpoints:
                    raise ValidationError(
                        f"created outpoint {op!r} already exists",
                        f"{loc}.created",
                        kind="duplicate",
                    )
                seen_created.add(op)
            for op in tx.inputs:
                if op in spent_in_block or op in self._confirmed_spent:
                    raise ValidationError(
                        f"outpoint {op!r} spent more than once across "
                        f"confirmed history and this block",
                        f"{loc}.tx.inputs",
                        kind="double_spend",
                    )
                spent_in_block.add(op)
            # Dependency order: explicit deps and creator prefixes must
            # resolve to an earlier block tx or an already-confirmed tx.
            # (A pool tx that is not being confirmed can never satisfy a
            # confirmed transaction's dependency.)
            for dep in tx.depends:
                if dep not in seen_ids and dep not in self._confirmed_ids:
                    raise MissingDependencyError(
                        f"confirmed transaction {tx.txid!r} depends on "
                        f"{dep!r}, which is neither confirmed nor an "
                        f"earlier block transaction",
                        missing=(dep,),
                        loc=f"{loc}.tx.depends",
                    )
            for op in tx.inputs:
                creator = outpoint_creator(op)
                if (
                    creator is not None
                    and creator != tx.txid
                    and creator not in seen_ids
                    and creator not in self._confirmed_ids
                    and creator in self._txs
                ):
                    raise MissingDependencyError(
                        f"confirmed transaction {tx.txid!r} spends "
                        f"{op!r}, whose creator is neither confirmed "
                        f"nor an earlier block transaction",
                        missing=(creator,),
                        loc=f"{loc}.tx.inputs",
                    )

    def _commit_confirm(self, items):
        confirmed_ids = [btx.tx.txid for btx in items]
        confirmed_set = set(confirmed_ids)
        pool_confirmed = set()
        conflict_removed = set()
        for btx in items:
            tx = btx.tx
            if tx.txid in self._txs:
                # Confirmation wins: removed by id, descendants kept.
                pool_confirmed.add(tx.txid)
                self._evict({tx.txid})
            conflicts = {
                self._outpoint_owner[op]
                for op in tx.inputs
                if op in self._outpoint_owner
            }
            conflicts -= confirmed_set
            if conflicts:
                removed = set()
                for cid in conflicts:
                    removed.add(cid)
                    removed.update(self._descendants.get(cid, frozenset()))
                removed -= confirmed_set
                if removed:
                    conflict_removed.update(removed)
                    self._evict(removed)
            self._confirmed_ids.add(tx.txid)
            for op in tx.inputs:
                self._confirmed_spent.add(op)
                self._confirmed_outpoints.discard(op)
            self._confirmed_outpoints.update(btx.created)
        return ConfirmResult(confirmed_ids, pool_confirmed, conflict_removed)

    def _evict(self, ids):
        for rid in ids:
            self._txs.pop(rid, None)
            self._parents.pop(rid, None)
            self._children.pop(rid, None)
        for op, owner in list(self._outpoint_owner.items()):
            if owner in ids:
                del self._outpoint_owner[op]
        # Rebuild edges so kept descendants of evicted parents lose the
        # dangling edge (their dependency is now a confirmed output).
        self._parents, self._children = build_edges(self._txs)
        self._refresh_closures()
