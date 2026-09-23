"""Independent exhaustive reference for package selection (test-only).

This reference deliberately does **not** import or call the project's
core solver/pool. It re-derives parent edges directly from raw ``Tx``
fields and enumerates all ``2**n`` subsets, so agreement with
``select_package`` is a genuine cross-check rather than a round trip.

It supports at most 16 transactions, matching the acceptance bound.
"""


def _creator(outpoint):
    i = outpoint.rfind(":")
    return outpoint[:i] if i > 0 else None


def brute_force_optimal(txs, block_weight):
    """Return ``(fee, sorted_ids, weight)`` for the optimum package.

    Ties are broken by the lexicographically smallest tuple of ids
    sorted ascending, independently implemented with plain tuples.
    """
    by_id = {t.txid: t for t in txs}
    parents = {}
    for t in txs:
        ps = set(t.depends)
        for op in t.inputs:
            c = _creator(op)
            if c is not None and c != t.txid and c in by_id:
                ps.add(c)
        parents[t.txid] = ps

    ids = sorted(by_id)
    n = len(ids)
    if n > 16:
        raise ValueError("reference supports at most 16 transactions")

    best_fee = -1
    best_tuple = None
    best_weight = None

    for bits in range(1 << n):
        chosen = {ids[k] for k in range(n) if (bits >> k) & 1}
        # ancestor closure, checked directly from the rebuilt edges
        ok = True
        for tid in chosen:
            if not parents[tid] <= chosen:
                ok = False
                break
        if not ok:
            continue
        w = sum(by_id[t].weight for t in chosen)
        if w > block_weight:
            continue
        f = sum(by_id[t].fee for t in chosen)
        tup = tuple(sorted(chosen))
        if f > best_fee or (f == best_fee and
                            (best_tuple is None or tup < best_tuple)):
            best_fee = f
            best_tuple = tup
            best_weight = w

    if best_tuple is None:  # only when even the empty set is infeasible
        best_fee, best_tuple, best_weight = 0, (), 0
    return best_fee, list(best_tuple), best_weight
