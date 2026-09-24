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
4. the atomic ancestor package -- an unordered 8-tx DAG submitted at
   once (the inclusive size boundary), accepted in topological order,
   followed by actually-triggered package validation errors (cycle,
   intra-package double spend, spend of a pool-owned outpoint, missing
   parent), each leaving the pool byte-for-byte unchanged.
"""

import time

from ancestor_pool import (
    AncestorPool,
    make_tx,
    select_package,
    ReplacementRejectedError,
    MissingDependencyError,
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

    hr("[5] Atomic ancestor package (at most 8 txs, order irrelevant)")
    pp = AncestorPool()
    pp.add(make_tx("base", weight=2, fee=1, inputs=("basecoin:0",)))
    # 8-tx DAG handed in deliberately non-topological order:
    # a -> b,c ; {b,c} -> d ; base(in pool) -> e -> f ; {d,f} -> g ;
    # h is independent. g is submitted first though it depends on d, f.
    package = [
        make_tx("g", 2, 2, inputs=("d:0", "f:0")),
        make_tx("h", 2, 2, inputs=("coin-h:0",)),
        make_tx("f", 2, 2, inputs=("e:0",)),
        make_tx("b", 2, 2, inputs=("a:0",)),
        make_tx("e", 2, 2, inputs=("base:0",)),
        make_tx("d", 2, 2, inputs=("b:0", "c:0")),
        make_tx("c", 2, 2, inputs=("a:1",)),
        make_tx("a", 2, 2, inputs=("coin-a:0",)),
    ]
    pkg = pp.add_package(package)
    print(f"accepted {len(pkg)} txs (the inclusive 8-tx boundary)")
    print(f"canonical order (parents first): {list(pkg.order)}")
    print(f"package totals: fee={pkg.added_fee} weight={pkg.added_weight}")
    print(f"pool size now: {len(pp)}; parents of g: {sorted(pp.parents('g'))}")
    print("edges of g: spends d:0 and f:0, both present in the package")

    # 9 txs must be rejected -- the boundary is inclusive at 8.
    nine = [make_tx(f"n{i}", 1, 0, inputs=(f"cn{i}:0",)) for i in range(9)]
    try:
        pp.add_package(nine)
    except ValidationError as err:
        print(f"9-tx package rejected: kind={err.kind!r} ({err})")

    print("-- real validation errors; pool must stay unchanged after each --")

    def expect_package_error(label, txs, expect_kind=None):
        before = sorted(pp.ids())
        try:
            pp.add_package(txs)
        except (ValidationError, MissingDependencyError) as err:
            kind = getattr(err, "kind", "missing_dependency")
            if expect_kind is not None:
                assert kind == expect_kind, (kind, expect_kind)
            after = sorted(pp.ids())
            print(f"  {label}: {type(err).__name__} kind={kind!r} "
                  f"loc={getattr(err, 'loc', '?')}")
            print(f"    pool unchanged: {before == after} ({len(before)} txs)")
            return
        raise AssertionError(f"{label}: expected an error")

    # Cycle inside the package (x depends on y; y spends x:0).
    expect_package_error(
        "cycle",
        [
            make_tx("x", 1, 1, inputs=("cx:0",), depends=("y",)),
            make_tx("y", 1, 1, inputs=("x:0",)),
        ],
        "cycle",
    )
    # Intra-package double spend of one outpoint.
    expect_package_error(
        "double spend",
        [
            make_tx("u", 1, 1, inputs=("q:0",)),
            make_tx("v", 1, 1, inputs=("q:0",)),
        ],
        "double_spend",
    )
    # Spending an outpoint already spent in the pool is never a
    # replacement via package submission (append-only).
    expect_package_error(
        "spends pool-owned outpoint",
        [make_tx("rbf", 1, 100, inputs=("basecoin:0",))],
        "conflict_with_pool",
    )
    # Explicit parent absent from both pool and package.
    expect_package_error(
        "missing parent",
        [make_tx("m", 1, 1, inputs=("cm:0",), depends=("ghost",))],
        "missing_dependency",
    )

    elapsed = time.perf_counter() - t0
    print()
    print(f"demo finished in {elapsed:.3f}s")


if __name__ == "__main__":
    main()
