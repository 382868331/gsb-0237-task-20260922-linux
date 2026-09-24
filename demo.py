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
   ``budget_exhausted`` instead of an optimality claim;
4. atomic dependency-package submission -- an 8-tx package submitted
   out of order, the 8-tx size boundary, and a real failure (a package
   double-spend / pool conflict) that leaves no half package.
"""

import time

from ancestor_pool import (
    AncestorPool,
    MAX_PACKAGE_SIZE,
    PackageConflictError,
    make_tx,
    select_package,
    ReplacementRejectedError,
    ValidationError,
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

    # ---- Atomic dependency-package submission ---------------------------
    hr("[5] Normal package: 8 dependent txs submitted out of order")
    # A fresh pool with one pre-existing ancestor.
    ppool = AncestorPool()
    ppool.add(make_tx("b0", 4, 3, inputs=("basecoin:0",)))
    # DAG (package nodes): a -> c,d ; {c,d} -> e ; e -> f,g ;
    # x and y are independent external-coin txs in the same package.
    # Hand-derived lex-smallest topological order:
    #   a, c, d, e, f, g, x, y
    package = [
        make_tx("f", 2, 4, inputs=("e:0",)),
        make_tx("y", 2, 1, inputs=("ex2:0",)),
        make_tx("a", 2, 5, inputs=("b0:0",)),
        make_tx("g", 2, 2, inputs=("e:1",)),
        make_tx("d", 2, 3, inputs=("a:1",)),
        make_tx("x", 2, 6, inputs=("ex1:0",)),
        make_tx("c", 2, 7, inputs=("a:0",)),
        make_tx("e", 2, 8, inputs=("c:0", "d:0",)),
    ]
    pres = ppool.submit_package(package)
    print("submission order (scrambled):", [t.txid for t in package])
    print("topological order committed  :", list(pres.order))
    print("attaches to existing ancestor:", sorted(pres.attached))
    print(f"package totals: fee={pres.total_fee} weight="
          f"{pres.total_weight}; pool size now {len(ppool)}")
    print("parents(e) =", sorted(ppool.parents("e")),
          " ancestors(f) =", sorted(ppool.ancestors("f")))

    hr("[6] Package size boundary: 9 txs refused, pool untouched")
    boundary_pool = AncestorPool()
    nine = [make_tx(f"n{i}", 1, 1, inputs=(f"nc{i}:0",)) for i in range(9)]
    try:
        boundary_pool.submit_package(nine)
    except ValidationError as err:
        print(f"ValidationError kind={err.kind!r}: {err}")
    print(f"limit is MAX_PACKAGE_SIZE={MAX_PACKAGE_SIZE}; pool size "
          f"after refusal: {len(boundary_pool)}")

    hr("[7] Package failure is atomic: pool-owned outpoint conflict")
    cpool = AncestorPool()
    cpool.add(make_tx("old", 10, 5, inputs=("owned:0",)))
    before = sorted(cpool.ids())
    # Even a vastly over-paying tx cannot evict via package submission.
    try:
        cpool.submit_package([
            make_tx("new", 1, 1_000_000, inputs=("owned:0",)),
            make_tx("tail", 1, 1, inputs=("new:0",)),
        ])
    except PackageConflictError as err:
        print(f"PackageConflictError: {err.txid!r} spends {err.outpoint!r} "
              f"owned by {err.owner!r}")
        print("  (package submission never replaces; use add())")
    print(f"pool unchanged after refusal: {sorted(cpool.ids()) == before} "
          f"{sorted(cpool.ids())}")
    # The legacy single-tx interface still does the fee-gated replacement.
    rres = cpool.add(make_tx("new", 1, 6, inputs=("owned:0",)))
    print(f"legacy add() still replaces: evicted={sorted(rres.removed_ids)} "
          f"pool={sorted(cpool.ids())}")

    elapsed = time.perf_counter() - t0
    print()
    print(f"demo finished in {elapsed:.3f}s")


if __name__ == "__main__":
    main()
