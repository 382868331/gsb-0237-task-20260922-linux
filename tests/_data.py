"""Deterministic synthetic transaction-DAG generator (test/demo only)."""

import random

from ancestor_pool import make_tx


def gen_txs(seed, n, *, max_weight=40, max_fee=50, max_parents=2):
    """Build a random but seeded DAG of ``n`` transactions.

    Edges come from spent creator outpoints (the explicit ``depends``
    field mirrors them). Every child spends a *unique* output of its
    parent, so generated batches never double-spend; each tx also
    consumes one unique external coin.
    """
    rng = random.Random(seed)
    txs = []
    for i in range(n):
        tid = f"t{i:02d}"
        inputs = [f"ext-{seed}-{i}:0"]
        parent_ids = []
        if i > 0 and max_parents > 0:
            k = rng.randrange(0, min(max_parents, i) + 1)
            for p in rng.sample(range(i), k):
                # vout = child index guarantees uniqueness across edges.
                inputs.append(f"t{p:02d}:{i}")
                parent_ids.append(f"t{p:02d}")
        txs.append(
            make_tx(
                tid,
                rng.randrange(1, max_weight + 1),
                rng.randrange(0, max_fee + 1),
                inputs=inputs,
                depends=tuple(sorted(parent_ids)),
            )
        )
    return txs
