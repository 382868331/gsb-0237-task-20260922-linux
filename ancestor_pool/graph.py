"""DAG construction for the task's transaction graph.

Edge convention (parent -> child)
---------------------------------
A transaction *u* is a parent of *v* when either:

1. ``v.depends`` explicitly names ``u.txid``; or
2. ``v`` spends an outpoint of the form ``"u.txid:vout"``. The creator is
   the substring before the **last** ``":"``; the edge is created only if
   a transaction with that id exists in the resolution universe. An
   outpoint whose prefix matches no known id is treated as an external
   coin (no edge, but it still participates in double-spend conflict
   detection).

Two transactions spending the *same* outpoint string conflict, but that
relationship is tracked separately (it is not a DAG edge).
"""

import heapq

from .errors import MissingDependencyError, ValidationError
from .model import validate_tx


def outpoint_creator(outpoint):
    """Return the creator-id prefix of ``"creator:vout"`` or ``None``."""
    idx = outpoint.rfind(":")
    if idx <= 0:
        return None
    return outpoint[:idx]


def validate_incoming_batch(incoming, pool_map, loc="txs"):
    """Structurally validate an incoming batch against an existing pool.

    Performs the same checks as :func:`validate_batch` over the union of
    pool and incoming transactions: duplicate ids (also against the
    pool), unresolvable explicit dependencies, and dependency cycles.
    Intra-batch double spends are rejected; spends of pool-owned
    outpoints are *allowed* (they are replacements) and handled by the
    pool layer.
    """
    by_id = {}
    intra_owners = set()
    for i, tx in enumerate(incoming):
        iloc = f"{loc}[{i}]"
        validate_tx(tx, iloc)
        if tx.txid in pool_map or tx.txid in by_id:
            raise ValidationError(
                f"txid {tx.txid!r} already exists in the batch/pool",
                iloc,
                kind="duplicate",
            )
        by_id[tx.txid] = tx
        for op in tx.inputs:
            if op in intra_owners:
                raise ValidationError(
                    f"outpoint {op!r} spent by two transactions in the "
                    f"same batch",
                    iloc,
                    kind="double_spend",
                )
            intra_owners.add(op)

    combined = dict(pool_map)
    combined.update(by_id)
    # Explicit dependencies of incoming txs must resolve somewhere.
    for i, tx in enumerate(incoming):
        for dep in tx.depends:
            if dep not in combined:
                raise MissingDependencyError(
                    f"transaction {tx.txid!r} depends on unknown id "
                    f"{dep!r}",
                    missing=(dep,),
                    loc=f"{loc}[{i}].depends",
                )
        for op in tx.inputs:
            if outpoint_creator(op) == tx.txid:
                raise ValidationError(
                    f"transaction {tx.txid!r} must not spend its own "
                    f"output {op!r}",
                    f"{loc}[{i}].inputs",
                    kind="cycle",
                )
    parents, _children = build_edges(combined)
    _check_acyclic(combined, parents, loc)
    return by_id


def validate_batch(txs, loc="txs"):
    """Validate an iterable of :class:`Tx` as a self-consistent batch.

    Returns a ``dict[txid -> Tx]``. Raises locatable errors on:
    non-Tx/field errors, duplicate ids, intra-batch double spends,
    unresolvable explicit dependencies, or dependency cycles.
    """
    if not isinstance(txs, (list, tuple)):
        # Accept any finite iterable, but materialise for indexing.
        if not hasattr(txs, "__iter__"):
            raise ValidationError(
                f"expected an iterable of Tx; got {type(txs).__name__}",
                loc,
                kind="type_error",
            )
        txs = list(txs)

    tx_map = {}
    outpoint_spenders = {}
    for i, tx in enumerate(txs):
        iloc = f"{loc}[{i}]"
        validate_tx(tx, iloc)
        if tx.txid in tx_map:
            raise ValidationError(
                f"duplicate txid {tx.txid!r} in batch",
                iloc,
                kind="duplicate",
            )
        tx_map[tx.txid] = tx
        for op in tx.inputs:
            owners = outpoint_spenders.setdefault(op, [])
            if owners:
                raise ValidationError(
                    f"outpoint {op!r} spent by both {owners[0]!r} and "
                    f"{tx.txid!r} in the same batch",
                    f"{iloc}.inputs",
                    kind="double_spend",
                )
            owners.append(tx.txid)
    _check_dependencies(tx_map, loc)
    parents, _children = build_edges(tx_map)
    _check_acyclic(tx_map, parents, loc)
    return tx_map


def _check_dependencies(tx_map, loc):
    # Only explicit ``depends`` must resolve inside a self-contained
    # batch. Spent outpoints whose "creator:vout" prefix matches no id
    # are external coins (see module docstring) and require nothing;
    # spending one's own output would be a self-cycle and is caught by
    # the acyclicity check below.
    for txid, tx in tx_map.items():
        for dep in tx.depends:
            if dep not in tx_map:
                raise MissingDependencyError(
                    f"transaction {txid!r} depends on unknown id {dep!r}",
                    missing=(dep,),
                    loc=f"{loc}[{txid}].depends",
                )
        for op in tx.inputs:
            if outpoint_creator(op) == txid:
                raise ValidationError(
                    f"transaction {txid!r} must not spend its own output "
                    f"{op!r}",
                    f"{loc}[{txid}].inputs",
                    kind="cycle",
                )


def build_edges(tx_map, base_ids=None):
    """Build ``(parents, children)`` id-sets for ``tx_map``.

    ``base_ids`` optionally extends the universe so edges to known ids
    outside ``tx_map`` are included; edges to unknown creators are
    omitted (external coins).
    """
    universe = set(tx_map)
    if base_ids:
        universe.update(base_ids)
    parents = {tid: set() for tid in tx_map}
    children = {tid: set() for tid in tx_map}
    for txid, tx in tx_map.items():
        for dep in tx.depends:
            if dep in universe:
                parents[txid].add(dep)
        for op in tx.inputs:
            creator = outpoint_creator(op)
            if (
                creator is not None
                and creator in universe
                and creator != txid
            ):
                parents[txid].add(creator)
    for child, ps in parents.items():
        for p in ps:
            if p in children:
                children[p].add(child)
    return parents, children


def _check_acyclic(tx_map, parents, loc):
    # Heap-based Kahn: smallest available id is always emitted next,
    # giving the lexicographically smallest topological order.
    indeg = {tid: 0 for tid in tx_map}
    children = {tid: set() for tid in tx_map}
    for tid, ps in parents.items():
        for p in ps:
            if p in indeg:
                indeg[tid] += 1
                children[p].add(tid)
    heap = [tid for tid, d in indeg.items() if d == 0]
    heapq.heapify(heap)
    seen = 0
    while heap:
        node = heapq.heappop(heap)
        seen += 1
        for ch in children[node]:
            indeg[ch] -= 1
            if indeg[ch] == 0:
                heapq.heappush(heap, ch)
    if seen != len(tx_map):
        cyclic = sorted(tid for tid, d in indeg.items() if d > 0)
        raise ValidationError(
            f"dependency cycle detected involving {cyclic}",
            loc,
            kind="cycle",
        )


def topo_order(tx_map, parents, children=None, subset=None):
    """Deterministic topological order.

    Ties are broken by smallest txid (heap-based Kahn), producing the
    lexicographically smallest topological order. When ``subset`` is
    given, the order is computed over the induced subgraph only.
    """
    if subset is None:
        nodes = set(tx_map)
    else:
        nodes = set(subset)
    if children is None:
        children = {tid: set() for tid in tx_map}
        for tid, ps in parents.items():
            for p in ps:
                if p in children:
                    children[p].add(tid)
    indeg = {}
    local_children = {tid: set() for tid in nodes}
    for tid in nodes:
        d = 0
        for p in parents.get(tid, ()):
            if p in nodes:
                d += 1
                local_children[p].add(tid)
        indeg[tid] = d
    heap = [tid for tid, d in indeg.items() if d == 0]
    heapq.heapify(heap)
    order = []
    while heap:
        node = heapq.heappop(heap)
        order.append(node)
        for ch in local_children[node]:
            indeg[ch] -= 1
            if indeg[ch] == 0:
                heapq.heappush(heap, ch)
    if len(order) != len(nodes):
        raise ValidationError(
            "dependency cycle detected during topological ordering",
            "?",
            kind="cycle",
        )
    return order


def closure_maps(txids, parents):
    """Return ``(ancestors, descendants)`` as ``dict[id -> frozenset]``."""
    ancestors = {}

    def anc(tid, stack):
        if tid in ancestors:
            return ancestors[tid]
        if tid in stack:
            raise ValidationError(
                f"dependency cycle detected at {tid!r}", "?", kind="cycle"
            )
        stack.add(tid)
        acc = set()
        for p in parents.get(tid, ()):  # parents may include base ids
            acc.add(p)
            acc.update(anc(p, stack))
        stack.discard(tid)
        result = frozenset(acc)
        ancestors[tid] = result
        return result

    for tid in txids:
        anc(tid, set())

    descendants = {tid: set() for tid in txids}
    for tid in txids:
        for a in ancestors[tid]:
            if a in descendants:
                descendants[a].add(tid)
    return ancestors, {tid: frozenset(s) for tid, s in descendants.items()}
