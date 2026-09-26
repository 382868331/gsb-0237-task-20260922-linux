"""Live demonstration: legacy select_package vs exact select_optimal.

Run:  python demo_iteration.py

Everything is computed live (no sleeps, no pre-recorded output); the
whole script finishes in well under 90 seconds. It prints:

1. both selection modes on the SAME pool, showing different results
   (the legacy interface breaks fee ties by id tuple; select_optimal
   breaks them by smaller total weight first);
2. ancestor closure: a low-fee ancestor pulled in by its high-fee
   child, and a shared ancestor paid for only once;
3. a mutually exclusive double-spend pair in a candidate list (a pool
   can never hold one), resolved by select_optimal;
4. the budget boundary: budget_exhausted returns a feasible package
   plus an admissible fee upper bound, never the "optimal" status.
"""

import time

from ancestor_pool import (
    AncestorPool,
    make_tx,
    select_optimal,
    select_package,
    STATUS_BUDGET_EXHAUSTED,
)


def hr(title):
    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


def show_optimal(r):
    print(f"status        : {r.status}  (nodes {r.nodes_used} "
          f"<= budget {r.node_budget})")
    print(f"selected topo : {list(r.selected)}")
    print(f"total fee     : {r.total_fee}")
    print(f"total weight  : {r.total_weight} / {r.block_weight}")
    print(f"fee upper bound: {r.fee_upper_bound}")
    print("ledger (txid: fee, weight -> cumulative):")
    for e in r.ledger:
        print(f"  {e.txid}: fee={e.fee} weight={e.weight} "
              f"-> fee {e.cumulative_fee}, weight {e.cumulative_weight}")


def main():
    t0 = time.perf_counter()

    hr("[1] Same pool, two modes, different tie-break outcomes")
    # "a" (heavy) and "b" (light) pay the same fee; only one fits.
    # Legacy select_package: id tuple wins -> "a".
    # select_optimal: smaller weight wins first -> "b".
    pool = AncestorPool()
    pool.add(make_tx("a", weight=6, fee=10, inputs=("ia:0",)))
    pool.add(make_tx("b", weight=4, fee=10, inputs=("ib:0",)))
    pool.add(make_tx("p", weight=10, fee=1, inputs=("coin:0",)))
    pool.add(make_tx("h", weight=10, fee=9, inputs=("p:0",)))
    print("pool:", sorted(pool.ids()))
    legacy = select_package(pool, 9, node_budget=200_000)
    print(f"legacy select_package (block 9): {list(legacy.selected)} "
          f"fee={legacy.total_fee} weight={legacy.total_weight} "
          f"({legacy.status})")
    exact = select_optimal(pool, 9, node_budget=200_000)
    print("select_optimal      (block 9):")
    show_optimal(exact)
    assert list(legacy.selected) != list(exact.selected)

    hr("[2] Ancestor closure: low-fee parent, high-fee child, shared root")
    pool2 = AncestorPool()
    pool2.add(make_tx("root", weight=10, fee=1, inputs=("c:0",)))
    pool2.add(make_tx("left", weight=10, fee=20, inputs=("root:0",)))
    pool2.add(make_tx("right", weight=10, fee=20, inputs=("root:1",)))
    pool2.add(make_tx("solo", weight=10, fee=15, inputs=("d:0",)))
    r2 = select_optimal(pool2, 30, node_budget=200_000)
    print("pool:", sorted(pool2.ids()), " block_weight=30")
    show_optimal(r2)
    print("note: root's weight is paid once for both children; "
          "the childless 'solo' loses to the shared-ancestor package")

    hr("[3] Mutually exclusive double spend (candidate list, not a pool)")
    # x and y spend the same coin and can never coexist; x's child c
    # cannot rescue it because y alone outbids the x+c package.
    candidates = [
        make_tx("x", weight=10, fee=5, inputs=("coin:0",)),
        make_tx("y", weight=10, fee=7, inputs=("coin:0",)),
        make_tx("c", weight=5, fee=1, inputs=("x:0",)),
        make_tx("z", weight=10, fee=1, inputs=("other:0",)),
    ]
    r3 = select_optimal(candidates, 100, node_budget=200_000)
    print("candidates:", sorted(t.txid for t in candidates),
          "(x and y double-spend coin:0)")
    show_optimal(r3)
    assert "x" not in r3.selected and "y" in r3.selected

    hr("[4] Budget boundary: budget_exhausted is not 'optimal'")
    import random
    rng = random.Random(20260926)
    big = AncestorPool()
    for i in range(24):
        parents = ()
        inputs = [f"ext{i}:0"]
        if i > 0 and rng.random() < 0.5:
            p = f"t{rng.randrange(i):02d}"
            inputs.append(f"{p}:{i}")
            parents = (p,)
        big.add(make_tx(f"t{i:02d}", rng.randrange(5, 30),
                        rng.randrange(1, 60), inputs=inputs,
                        depends=parents))
    bounded = select_optimal(big, 200, node_budget=5)
    print(f"pool of {len(big)} txs, block_weight=200, node_budget=5")
    show_optimal(bounded)
    assert bounded.status == STATUS_BUDGET_EXHAUSTED
    proven = select_optimal(big, 200, node_budget=50_000_000)
    print(f"re-run with generous budget: status={proven.status} "
          f"fee={proven.total_fee} (nodes {proven.nodes_used})")
    print(f"check: exhausted upper bound {bounded.fee_upper_bound} "
          f">= proven optimum {proven.total_fee}: "
          f"{bounded.fee_upper_bound >= proven.total_fee}")
    print(f"pool unchanged by selection: {len(big)} txs still present")

    elapsed = time.perf_counter() - t0
    print()
    print(f"demo finished in {elapsed:.3f}s (live computation, no sleeps)")


if __name__ == "__main__":
    main()
