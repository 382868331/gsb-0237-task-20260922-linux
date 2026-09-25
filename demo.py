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
4. block confirmation -- a confirmed parent leaves the pool while its
   child stays, re-pointed at the confirmed outpoint; an out-of-block
   conflict evicts a conflicting tx and its descendants; an illegal
   block rolls back wholesale.
"""

import time

from ancestor_pool import (
    AncestorPool,
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

    hr("[5] Block confirmation: parent confirmed, child retained")
    # Fresh DAG: coin:0 -> CP -> CC -> CG, plus independent CK.
    cpool = AncestorPool()
    cpool.add(make_tx("CP", weight=40, fee=3, inputs=("ccoin:0",)))
    cpool.add(make_tx("CC", weight=10, fee=40, inputs=("CP:0",)))
    cpool.add(make_tx("CG", weight=8, fee=5, inputs=("CC:0",)))
    cpool.add(make_tx("CK", weight=12, fee=7, inputs=("ckoin:0",)))
    print("pool before:", sorted(cpool.ids()))
    # One block confirms CP (same id) and a rival RX spending CP:0.
    # The pool spender of CP:0 is CC (different id), so CC and its
    # descendant CG are evicted as conflict losers; CP leaves as
    # confirmed; CK is untouched.
    res = cpool.confirm([
        make_tx("CP", weight=40, fee=3, inputs=("ccoin:0",)),
        make_tx("RX", weight=10, fee=60, inputs=("CP:0",)),
    ])
    print("confirmed   :", list(res.confirmed_ids))
    print("evicted     :", sorted(res.evicted_ids))
    print("pool after  :", sorted(cpool.ids()))
    print("confirmed outpoints:", sorted(cpool.confirmed_outpoints()))

    # Parent-confirmed / child-retained case in isolation.
    ppool = AncestorPool()
    ppool.add(make_tx("PP", weight=30, fee=4, inputs=("pcoin:0",)))
    ppool.add(make_tx("PC", weight=6, fee=50, inputs=("PP:0",)))
    cres = ppool.confirm([make_tx("PP", 30, 4, inputs=("pcoin:0",))])
    print(f"child retained: pool={sorted(ppool.ids())}, "
          f"reparented={sorted(cres.reparented_ids)}, "
          f"confirmed parents of PC={sorted(ppool.confirmed_parents('PC'))}")
    csel = select_package(ppool, 100, node_budget=10_000)
    print(f"select after confirm: {list(csel.selected)} "
          f"fee={csel.total_fee} weight={csel.total_weight} "
          "(confirmed PP contributes no weight/fee)")

    hr("[6] Illegal confirmation rolls the whole batch back")
    bad_pool = AncestorPool()
    bad_pool.add(make_tx("CP", weight=40, fee=3, inputs=("ccoin:0",)))
    bad_pool.add(make_tx("CC", weight=10, fee=40, inputs=("CP:0",)))
    try:
        # Child listed before its parent: dependency-order violation.
        bad_pool.confirm([
            make_tx("CC", weight=10, fee=40, inputs=("CP:0",)),
            make_tx("CP", weight=40, fee=3, inputs=("ccoin:0",)),
        ])
    except Exception as err:  # ConfirmRejectedError
        print(f"{type(err).__name__}: {err.reason}")
        print(f"  loc   : {err.loc}")
        print(f"  detail: {err}")
    print("pool unchanged after rollback:", sorted(bad_pool.ids()))

    elapsed = time.perf_counter() - t0
    print()
    print(f"demo finished in {elapsed:.3f}s")


if __name__ == "__main__":
    main()
