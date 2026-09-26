"""Independent exhaustive reference for select_optimal (test-only).

This reference deliberately does NOT import or call the project's core
solver, pool, or graph modules. It re-derives parent edges and
double-spend conflicts directly from raw Tx fields and enumerates all
2**n subsets, so agreement with select_optimal is a genuine
cross-check rather than a round trip.

It supports at most 16 transactions, matching the acceptance bound for
the small-pool exhaustive comparison.
"""


def _creator(outpoint):
    i = outpoint.rfind(":")
    return outpoint[:i] if i > 0 else None


def brute_force_conflict_optimal(txs, block_weight):
    """Return (fee, weight, sorted_ids) for the optimum package.

    Feasible subsets are ancestor-closed, conflict-free (no two txs
    spending the same outpoint), and within the weight cap. Ties:
    higher fee, then lower total weight, then the lexicographically
    smallest ascending-id tuple -- implemented independently with
    plain tuples.
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

    spenders = {}
    for t in txs:
        for op in t.inputs:
            spenders.setdefault(op, set()).add(t.txid)

    ids = sorted(by_id)
    n = len(ids)
    if n > 16:
        raise ValueError("reference supports at most 16 transactions")

    best = None  # (fee, -weight, reversed-tuple) compared manually

    for bits in range(1 << n):
        chosen = {ids[k] for k in range(n) if (bits >> k) & 1}
        ok = True
        for tid in chosen:
            if not parents[tid] <= chosen:
                ok = False
                break
        if not ok:
            continue
        seen_ops = set()
        for tid in chosen:
            for op in by_id[tid].inputs:
                if op in seen_ops:
                    ok = False
                    break
                seen_ops.add(op)
            if not ok:
                break
        if not ok:
            continue
        w = sum(by_id[t].weight for t in chosen)
        if w > block_weight:
            continue
        f = sum(by_id[t].fee for t in chosen)
        tup = tuple(sorted(chosen))
        if (
            best is None
            or f > best[0]
            or (f == best[0] and w < best[1])
            or (f == best[0] and w == best[1] and tup < best[2])
        ):
            best = (f, w, tup)

    if best is None:  # only the empty set is feasible
        best = (0, 0, ())
    return best[0], best[1], list(best[2])
