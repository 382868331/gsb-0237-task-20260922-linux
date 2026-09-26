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


def gen_txs_with_conflicts(seed, n, *, max_weight=40, max_fee=50,
                           max_parents=2, conflict_every=5):
    """Seeded DAG like gen_txs but with occasional double-spend pairs.

    Every conflict_every-th transaction re-spends the external coin of
    an earlier independent transaction, creating a mutually exclusive
    pair that a conflict-aware selector must not combine. Batches from
    this generator cannot be added to an AncestorPool (the pool rejects
    intra-batch double spends); they are candidate lists for
    select_optimal.
    """
    rng = random.Random(seed)
    txs = []
    externals = []
    for i in range(n):
        tid = f"t{i:02d}"
        if (
            conflict_every > 0
            and i > 0
            and i % conflict_every == 0
            and externals
        ):
            inputs = [rng.choice(externals)]
        else:
            inputs = [f"ext-{seed}-{i}:0"]
            externals.append(inputs[0])
        parent_ids = []
        if i > 0 and max_parents > 0:
            k = rng.randrange(0, min(max_parents, i) + 1)
            for p in rng.sample(range(i), k):
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
