"""Live demonstration of the new ``select_optimal`` mode.

Run:  python demo_iteration.py

Everything is computed live in well under 90 seconds (no sleeps, no
pre-recorded answers). It prints, on one shared synthetic set-up:

1. the old fast selector (``select_package``) and the new exact
   selector (``select_optimal``) disagreeing on the *same* pool because
   of their different documented tie-breaks;
2. the three required structural cases on raw candidate lists (which,
   unlike a pool, may hold double spenders): a low-fee ancestor with a
   high-fee child, a shared ancestor paid only once, and mutually
   exclusive double spends;
3. the budget boundary: a tiny node budget returns a feasible package
   plus an admissible fee upper bound, honestly labelled
   ``budget_exhausted`` -- never ``optimal``;
4. proof that selection is read-only: the pool is unchanged and
   nothing is marked confirmed.
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


def show(result, cap):
    print(f"  status      : {result.status}")
    print(f"  topological : {list(result.selected)}")
    print(f"  ledger      : fee={result.total_fee} "
          f"weight={result.total_weight}/{cap}")
    print(f"  fee upper bound: {result.fee_upper_bound} "
          f"(nodes {result.nodes_used} <= budget {result.node_budget})")


def main():
    t0 = time.perf_counter()

    hr("[1] Same pool, two modes, different tie-breaks")
    # Equal fee, different weight: the old selector keeps its
    # documented ascending-id-tuple tie-break; select_optimal prefers
    # the smaller total weight first.
    pool = AncestorPool()
    pool.add(make_tx("a", weight=20, fee=10, inputs=("x:0",)))
    pool.add(make_tx("b", weight=10, fee=10, inputs=("y:0",)))
    print("pool:", sorted(pool.ids()),
          "(a: w20 f10, b: w10 f10, block_weight=25)")
    old = select_package(pool, 25, node_budget=200_000)
    print("old select_package (id-tuple tie-break):")
    print(f"  -> {list(old.selected)} fee={old.total_fee} "
          f"weight={old.total_weight} status={old.status}")
    new = select_optimal(pool, 25, node_budget=200_000)
    print("new select_optimal (smaller weight first):")
    show(new, 25)
    assert list(old.selected) != list(new.selected)

    hr("[2] Low-fee ancestor + high-fee child (not greedy-then-backfill)")
    # A pure fee-rate greedy that backfills ancestors is not asked to
    # prove anything; select_optimal exactly compares {p,h} against
    # the rival k under the cap.
    txs = [
        make_tx("p", 10, 1, inputs=("c:0",)),     # low-fee ancestor
        make_tx("h", 10, 100, inputs=("p:0",)),   # high-fee child
        make_tx("k", 15, 90, inputs=("d:0",)),    # independent rival
    ]
    print("candidates: p(w10 f1) <- h(w10 f100); k(w15 f90); cap 20")
    show(select_optimal(txs, 20, node_budget=10_000), 20)

    hr("[3] Shared ancestor paid only once")
    txs = [
        make_tx("root", 10, 1, inputs=("c:0",)),
        make_tx("c1", 10, 60, inputs=("root:0",)),
        make_tx("c2", 10, 60, inputs=("root:1",)),
        make_tx("solo", 25, 100, inputs=("d:0",)),
    ]
    print("candidates: root(w10 f1) <- c1,c2 (w10 f60 each); "
          "solo(w25 f100); cap 30")
    show(select_optimal(txs, 30, node_budget=10_000), 30)

    hr("[4] Mutually exclusive double spends")
    # Raw candidate lists may contain conflicts a pool never would;
    # both spend coin:0 and cannot coexist even though both fit.
    txs = [
        make_tx("p", 10, 1, inputs=("coin:0",)),
        make_tx("h", 10, 60, inputs=("p:0",)),
        make_tx("r", 10, 70, inputs=("coin:0",)),  # rival spend
    ]
    print("candidates: p(w10 f1) <- h(w10 f60); r(w10 f70) spends the "
          "same coin:0; cap 100 (everything would fit)")
    show(select_optimal(txs, 100, node_budget=10_000), 100)

    hr("[5] Budget boundary: feasible answer + upper bound, not 'optimal'")
    pool2 = AncestorPool()
    pool2.add(make_tx("p", 10, 1, inputs=("c:0",)))
    pool2.add(make_tx("h", 10, 9, inputs=("p:0",)))
    pool2.add(make_tx("k", 10, 8, inputs=("d:0",)))
    pool2.add(make_tx("q", 10, 7, inputs=("e:0",)))
    bounded = select_optimal(pool2, 25, node_budget=1)
    print("node_budget=1 on a 4-tx pool:")
    show(bounded, 25)
    assert bounded.status == STATUS_BUDGET_EXHAUSTED
    proven = select_optimal(pool2, 25, node_budget=1_000_000)
    print(f"with a generous budget the proven optimum is "
          f"fee={proven.total_fee} {list(proven.selected)}; the "
          f"exhausted run's upper bound {bounded.fee_upper_bound} "
          f"is admissible (>= {proven.total_fee})")

    hr("[6] Pool state is read-only during selection")
    before = sorted(pool2.ids())
    select_optimal(pool2, 25, node_budget=1_000_000)
    print(f"pool before: {before}")
    print(f"pool after : {sorted(pool2.ids())}  (unchanged, nothing "
          f"marked confirmed)")
    assert sorted(pool2.ids()) == before

    elapsed = time.perf_counter() - t0
    print()
    print(f"demo_iteration finished in {elapsed:.3f}s (limit 90s)")


if __name__ == "__main__":
    main()
