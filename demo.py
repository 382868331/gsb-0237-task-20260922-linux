"""Fixed-input demonstration for the ancestor-package replacement pool.

Run:  python demo.py

Everything is computed live (no sleeps, no pre-recorded output). It
prints:

1. a normal path -- a conflicting replacement that evicts a low-fee
   parent and its high-fee descendant, followed by a fee-optimal,
   dependency-closed block package with per-exclusion reasons;
2. an actually-triggered rejection boundary -- a replacement whose fee
   rate is not strictly higher is refused atomically;
3. the search-budget boundary -- a tiny node budget returns
   ``budget_exhausted`` instead of an optimality claim.
4. a block confirmation -- a confirmed parent leaves the pool while
   its child stays (now depending on a confirmed output), and a
   conflicting pool subtree is evicted.
"""

import time

from ancestor_pool import (
    AncestorPool,
    BlockTx,
    make_tx,
    select_package,
    ReplacementRejectedError,
    STATUS_BUDGET_EXHAUSTED,
)


def hr(title):
    print()
    print("=" * 68)
    print(title)
    print("=" * 68)


def main():
    t0 = time.perf_counter()

    # ---- Deterministic synthetic DAG -------------------------------------
    # coin:0 -> P (low-fee parent) -> H (high-fee child)
    # independent rival K spends a different coin.
    # L (heavy mid-fee) -> D (its child); O cannot fit in any block.
    pool = AncestorPool()
    pool.add(make_tx("P", weight=40, fee=2, inputs=("coin:0",)))
    pool.add(make_tx("H", weight=10, fee=90, inputs=("P:0",)))
    pool.add(make_tx("K", weight=20, fee=30, inputs=("other:0",)))
    pool.add(make_tx("L", weight=40, fee=100, inputs=("coinL:0",)))
    pool.add(make_tx("D", weight=5, fee=0, inputs=("L:0",)))
    pool.add(make_tx("O", weight=90, fee=1000, inputs=("coinO:0",)))

    hr("[1] Normal path: conflicting replacement")
    print("pool before:", sorted(pool.ids()))
    # R spends coin:0 -> conflicts with P, and H (descendant) is an
    # orphan and is evicted too. Fee 200 > 92 and rate 200/20 > 92/50.
    result = pool.add(make_tx("R", weight=20, fee=200, inputs=("coin:0",)))
    print(f"accepted R; evicted {sorted(result.removed_ids)} "
          f"(removed fee {result.removed_fee})")
    print("pool after: ", sorted(pool.ids()))

    hr("[2] Fee-optimal dependency-closed package (block_weight=45)")
    sel = select_package(pool, 45, node_budget=200_000)
    print(f"status      : {sel.status}  (nodes used {sel.nodes_used} "
          f"<= budget {sel.node_budget})")
    print(f"topological : {list(sel.selected)}")
    print(f"total fee   : {sel.total_fee}")
    print(f"total weight: {sel.total_weight} / {sel.block_weight}")
    print("excluded transactions (one locatable reason each):")
    for ex in sel.excluded:
        print(f"  - {ex.txid}: {ex.code}  ({ex.detail})")

    hr("[3] Actually-triggered rejection boundary")
    # S spends other:0 -> conflicts with K (fee 30, weight 20, rate 1.5).
    # Fee 31 passes the absolute-fee rule, but rate 31/25 = 1.24 < 1.5,
    # so the replacement is refused and the pool is untouched.
    try:
        pool.add(make_tx("S", weight=25, fee=31, inputs=("other:0",)))
    except ReplacementRejectedError as err:
        print(f"ReplacementRejectedError: {err.reason}")
        print(f"  incoming : {err.incoming_id}")
        print(f"  evicted-would-be: {sorted(err.removed_ids)}")
        print(f"  detail   : {err.detail}")
    print(f"pool unchanged after refusal: {sorted(pool.ids())}")

    hr("[4] Budget boundary: budget_exhausted is not an optimality proof")
    bounded = select_package(pool, 45, node_budget=2)
    print(f"status      : {bounded.status}")
    print(f"nodes used  : {bounded.nodes_used} (budget 2)")
    print(f"best package found: {list(bounded.selected)} "
          f"fee={bounded.total_fee} -- returned without optimality claim")
    assert bounded.status == STATUS_BUDGET_EXHAUSTED

    hr("[5] Block confirmation: descendants of confirmed txs survive")
    # Fresh pool: A -> B chain (A will be confirmed) plus an unrelated
    # C -> D subtree (C will be double-spent by a block transaction).
    pool2 = AncestorPool()
    a = make_tx("A", weight=10, fee=40, inputs=("coinA:0",))
    b = make_tx("B", weight=10, fee=30, inputs=("A:0",), depends=("A",))
    c = make_tx("C", weight=10, fee=5, inputs=("coinC:0",))
    d = make_tx("D", weight=10, fee=9, inputs=("C:0",), depends=("C",))
    pool2.add_many([a, b, c, d])
    print(f"pool before block: {sorted(pool2.ids())}")

    # The ledger confirms A (creating A:0) and a rival W that
    # double-spends C's input: A leaves the pool but B stays (now
    # depending on the confirmed output A:0); C and its descendant D
    # are evicted together.
    w = make_tx("W", weight=10, fee=5, inputs=("coinC:0",))
    block = [BlockTx(a, created=("A:0",)), BlockTx(w, created=("W:0",))]
    result = pool2.confirm(block)
    print(f"confirmed          : {list(result.confirmed_ids)}")
    print(f"removed by confirm : {sorted(result.pool_confirmed_ids)}")
    print(f"removed by conflict: {sorted(result.conflict_removed_ids)}")
    print(f"pool after block   : {sorted(pool2.ids())}  "
          f"(B kept, now depends on confirmed output A:0)")
    print(f"confirmed outpoints: {sorted(pool2.confirmed_outpoints())}")
    sel = select_package(pool2, 10)
    print(f"select(weight=10)  : {list(sel.selected)} "
          f"fee={sel.total_fee} -- confirmed A contributes nothing")

    # coinA:0 is now confirmed-spent, so add() rejects re-spending it.
    try:
        pool2.add(make_tx("R3", weight=5, fee=100, inputs=("coinA:0",)))
    except Exception as err:
        print(f"add spending confirmed-spent coin rejected: "
              f"{type(err).__name__}: {err}")

    elapsed = time.perf_counter() - t0
    print()
    print(f"demo finished in {elapsed:.3f}s")


if __name__ == "__main__":
    main()
