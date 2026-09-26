"""Independent exhaustive reference for ``select_optimal`` (test-only).

Like ``tests/_reference.py`` for the old selector, this reference
deliberately does **not** import or call the core solver. It re-derives
parent edges and conflict pairs directly from raw ``Tx`` fields and
enumerates all ``2**n`` subsets, applying the new contract:

* ancestor closure over the candidate set;
* conflicting transactions (same spent outpoint string) never coexist;
* objective: max total fee, then min total weight, then the
  lexicographically smallest ascending-id tuple.

Supports at most 16 candidates so the enumeration stays instant.
"""


def _creator(outpoint):
    i = outpoint.rfind(":")
    return outpoint[:i] if i > 0 else None


def brute_force_optimal_v2(txs, block_weight):
    """Return ``(fee, weight, sorted_ids)`` for the optimal package."""
    by_id = {t.txid: t for t in txs}
    if len(by_id) != len(txs):
        raise ValueError("duplicate ids in reference input")
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
        raise ValueError("reference supports at most 16 candidates")

    best = (0, 0, ())  # fee, weight, ascending-id tuple
    for bits in range(1 << n):
        chosen = {ids[k] for k in range(n) if (bits >> k) & 1}
        if not chosen:
            continue
        # ancestor closure
        if any(not parents[t] <= chosen for t in chosen):
            continue
        # conflict exclusion: no outpoint spent by two chosen txs
        seen = set()
        if any(op in seen or seen.add(op) for t in chosen for op in by_id[t].inputs):
            continue
        w = sum(by_id[t].weight for t in chosen)
        if w > block_weight:
            continue
        f = sum(by_id[t].fee for t in chosen)
        tup = tuple(sorted(chosen))
        if (
            f > best[0]
            or (f == best[0] and w < best[1])
            or (f == best[0] and w == best[1] and tup < best[2])
        ):
            best = (f, w, tup)
    return best[0], best[1], list(best[2])
