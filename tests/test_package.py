"""Tests for atomic dependency-package submission (submit_package).

Expected values are hand-computed / analytic (parent edges re-derived
directly from raw Tx fields, exactly like the independent selection
reference). The core implementation is never used to generate expected
values.
"""

import unittest

from ancestor_pool import (
    AncestorPool,
    MAX_PACKAGE_SIZE,
    PackageConflictError,
    Tx,
    make_tx,
    ValidationError,
    MissingDependencyError,
    select_package,
    STATUS_PROVEN_OPTIMAL,
)


def _raw_parents(tx, universe):
    """Re-derive parent ids from raw fields without touching pool state."""
    ps = set(tx.depends)
    for op in tx.inputs:
        i = op.rfind(":")
        creator = op[:i] if i > 0 else None
        if creator is not None and creator != tx.txid and creator in universe:
            ps.add(creator)
    return ps


def _pool_snapshot(pool):
    txs, parents = pool.edge_maps()
    return (
        sorted(txs),
        {k: sorted(v) for k, v in parents.items()},
        dict(pool._outpoint_owner),  # noqa: SLF001 - white-box atomicity check
    )


class TestPackageHappyPath(unittest.TestCase):
    def setUp(self):
        self.pool = AncestorPool()
        # Pre-existing ancestor the package will attach to.
        self.pool.add(make_tx("P", 10, 4, inputs=("coinP:0",)))

    def _scrambled_package(self):
        # DAG: P -> C -> G -> T ; C -> S -> T (diamond).
        # Submitted deliberately out of order.
        return [
            make_tx("T", 10, 20, inputs=("G:0", "S:0")),
            make_tx("C", 20, 9, inputs=("P:0",)),
            make_tx("S", 5, 3, inputs=("C:1",)),
            make_tx("G", 5, 7, inputs=("C:0",)),
        ]

    def test_arbitrary_order_committed_as_dag(self):
        result = self.pool.submit_package(self._scrambled_package())
        # All four present alongside the pre-existing ancestor.
        self.assertEqual(set(self.pool.ids()), {"P", "C", "G", "S", "T"})
        # Lexicographically-smallest topological order, hand-derived:
        # Kahn: start {C}; then heap {G,S} -> G; then S; then T.
        self.assertEqual(result.order, ("C", "G", "S", "T"))
        self.assertEqual(set(result.txids), {"C", "G", "S", "T"})
        self.assertEqual(result.attached, frozenset({"P"}))
        self.assertEqual(result.total_fee, 9 + 7 + 3 + 20)
        self.assertEqual(result.total_weight, 20 + 5 + 5 + 10)

    def test_edges_and_closures_after_commit(self):
        self.pool.submit_package(self._scrambled_package())
        self.assertEqual(set(self.pool.parents("C")), {"P"})
        self.assertEqual(set(self.pool.parents("G")), {"C"})
        self.assertEqual(set(self.pool.parents("S")), {"C"})
        self.assertEqual(set(self.pool.parents("T")), {"G", "S"})
        # Closures are consistent with independently re-derived edges.
        universe = set(self.pool.ids())
        for tid in universe:
            raw = _raw_parents(self.pool.get(tid), universe)
            self.assertEqual(set(self.pool.parents(tid)), raw, tid)
        self.assertEqual(
            set(self.pool.ancestors("T")), {"P", "C", "G", "S"}
        )
        self.assertEqual(set(self.pool.descendants("P")), {"C", "G", "S", "T"})
        # Outpoint ownership registered for every package input.
        self.assertEqual(self.pool.spender_of("C:0"), "G")
        self.assertEqual(self.pool.spender_of("C:1"), "S")

    def test_explicit_depends_attaches_to_pool_ancestor(self):
        # Parent edge only through explicit depends (no creator outpoint).
        result = self.pool.submit_package([
            make_tx("D", 5, 6, inputs=("fresh:0",), depends=("P",)),
        ])
        self.assertEqual(result.order, ("D",))
        self.assertEqual(result.attached, frozenset({"P"}))
        self.assertEqual(set(self.pool.parents("D")), {"P"})

    def test_unknown_creator_outpoint_is_external_coin(self):
        # Per the fixed contract, "ghost:0" with no matching creator is
        # an external coin: no parent edge, no missing-dependency error
        # (same semantics as single-tx add).
        result = self.pool.submit_package([
            make_tx("c", 5, 1, inputs=("ghost:0",)),
        ])
        self.assertEqual(result.attached, frozenset())
        self.assertEqual(set(self.pool.parents("c")), set())
        self.assertEqual(self.pool.spender_of("ghost:0"), "c")

    def test_single_transaction_package(self):
        result = self.pool.submit_package(
            [make_tx("X", 3, 2, inputs=("x:0",))]
        )
        self.assertEqual(result.order, ("X",))
        self.assertEqual(result.attached, frozenset())
        self.assertIn("X", self.pool)

    def test_accepts_generator_iterable(self):
        gen = (make_tx(f"g{i}", 1, 1, inputs=(f"gc{i}:0",)) for i in range(3))
        self.pool.submit_package(gen)
        self.assertEqual(
            set(self.pool.ids()), {"P", "g0", "g1", "g2"}
        )

    def test_then_selectable_as_one_closed_package(self):
        self.pool.submit_package(self._scrambled_package())
        sel = select_package(self.pool, 45, node_budget=100_000)
        # Closure forces P with the chain; hand check: {P,C,G,S,T}
        # weighs 10+20+5+5+10 = 50 > 45, so T is unreachable. Best
        # closed prefix: P+C+G+S = weight 40 fee 23.
        self.assertEqual(sel.status, STATUS_PROVEN_OPTIMAL)
        self.assertEqual(list(sel.selected), ["P", "C", "G", "S"])
        self.assertEqual(sel.total_fee, 4 + 9 + 7 + 3)


class TestPackageBoundaries(unittest.TestCase):
    def _independent(self, n):
        return [
            make_tx(f"t{i:02d}", 1, 1, inputs=(f"ext{i:02d}:0",))
            for i in range(n)
        ]

    def test_exactly_eight_accepted(self):
        pool = AncestorPool()
        result = pool.submit_package(self._independent(MAX_PACKAGE_SIZE))
        self.assertEqual(len(result.order), 8)
        self.assertEqual(len(pool), 8)

    def test_nine_rejected_without_change(self):
        pool = AncestorPool()
        before = _pool_snapshot(pool)
        with self.assertRaises(ValidationError) as cm:
            pool.submit_package(self._independent(MAX_PACKAGE_SIZE + 1))
        self.assertEqual(cm.exception.kind, "package_too_large")
        self.assertEqual(_pool_snapshot(pool), before)
        self.assertEqual(len(pool), 0)

    def test_empty_package_rejected(self):
        pool = AncestorPool()
        with self.assertRaises(ValidationError) as cm:
            pool.submit_package([])
        self.assertEqual(cm.exception.kind, "empty")
        self.assertEqual(len(pool), 0)

    def test_bad_iterable_type(self):
        pool = AncestorPool()
        with self.assertRaises(ValidationError):
            pool.submit_package("not-a-tx")
        with self.assertRaises(ValidationError):
            pool.submit_package(123)

    def test_capacity_not_exceeded(self):
        pool = AncestorPool()
        # Fill to 27.
        pool.add_many(
            [make_tx(f"e{i:02d}", 1, 0, inputs=(f"e-coin{i:02d}:0",))
             for i in range(27)]
        )
        before = _pool_snapshot(pool)
        # 27 + 4 = 31 > 30: rejected (packages never evict).
        with self.assertRaises(ValidationError) as cm:
            pool.submit_package(self._independent(4))
        self.assertEqual(cm.exception.kind, "capacity")
        self.assertEqual(_pool_snapshot(pool), before)
        self.assertEqual(len(pool), 27)
        # 27 + 3 = 30 fits exactly.
        pool.submit_package(
            [make_tx(f"f{i:02d}", 1, 0, inputs=(f"f-coin{i:02d}:0",))
             for i in range(3)]
        )
        self.assertEqual(len(pool), 30)


class TestPackageFailuresAtomic(unittest.TestCase):
    def setUp(self):
        self.pool = AncestorPool()
        self.pool.add(make_tx("P", 10, 4, inputs=("coinP:0",)))
        self.pool.add(make_tx("O", 10, 5, inputs=("owned:0",)))

    def test_cycle_inside_package_rejected(self):
        before = _pool_snapshot(self.pool)
        # b depends on a while a spends b:0 -> internal cycle.
        a = make_tx("a", 5, 1, inputs=("b:0",))
        b = make_tx("b", 5, 1, inputs=("ext:0",), depends=("a",))
        with self.assertRaises(ValidationError) as cm:
            self.pool.submit_package([a, b])
        self.assertEqual(cm.exception.kind, "cycle")
        self.assertEqual(_pool_snapshot(self.pool), before)
        self.assertNotIn("a", self.pool)
        self.assertNotIn("b", self.pool)

    def test_intra_package_double_spend_rejected(self):
        before = _pool_snapshot(self.pool)
        with self.assertRaises(ValidationError) as cm:
            self.pool.submit_package([
                make_tx("a", 5, 1, inputs=("shared:0",)),
                make_tx("b", 5, 1, inputs=("P:0", "shared:0",)),
            ])
        self.assertEqual(cm.exception.kind, "double_spend")
        self.assertEqual(_pool_snapshot(self.pool), before)
        # The first tx's input must not have been registered.
        self.assertIsNone(self.pool.spender_of("shared:0"))

    def test_missing_explicit_dependency_rejected(self):
        before = _pool_snapshot(self.pool)
        with self.assertRaises(MissingDependencyError):
            self.pool.submit_package([
                make_tx("c", 5, 1, depends=("nobody",)),
            ])
        self.assertEqual(_pool_snapshot(self.pool), before)

    def test_pool_outpoint_conflict_is_not_a_replacement(self):
        before = _pool_snapshot(self.pool)
        # A package tx spending O's outpoint must be refused, never
        # evict O (regardless of how high its fee is).
        with self.assertRaises(PackageConflictError) as cm:
            self.pool.submit_package([
                make_tx("r", 1, 10_000, inputs=("owned:0",)),
            ])
        self.assertEqual(cm.exception.txid, "r")
        self.assertEqual(cm.exception.outpoint, "owned:0")
        self.assertEqual(cm.exception.owner, "O")
        self.assertEqual(_pool_snapshot(self.pool), before)
        self.assertIn("O", self.pool)
        self.assertIsNone(self.pool.spender_of("r-nothing"))
        # The legacy single-tx path still performs the replacement.
        res = self.pool.add(make_tx("r", 1, 6, inputs=("owned:0",)))
        self.assertTrue(res.replaced)
        self.assertNotIn("O", self.pool)

    def test_duplicate_id_within_package(self):
        before = _pool_snapshot(self.pool)
        with self.assertRaises(ValidationError) as cm:
            self.pool.submit_package([
                make_tx("dup", 1, 1, inputs=("d1:0",)),
                make_tx("dup", 1, 1, inputs=("d2:0",)),
            ])
        self.assertEqual(cm.exception.kind, "duplicate")
        self.assertEqual(_pool_snapshot(self.pool), before)

    def test_id_already_in_pool(self):
        before = _pool_snapshot(self.pool)
        with self.assertRaises(ValidationError) as cm:
            self.pool.submit_package([
                make_tx("P", 1, 1, inputs=("p2:0",)),
            ])
        self.assertEqual(cm.exception.kind, "duplicate")
        self.assertEqual(_pool_snapshot(self.pool), before)

    def test_invalid_field_locates_package_index(self):
        before = _pool_snapshot(self.pool)
        # Build the bad Tx directly: make_tx validates eagerly, so use
        # the raw constructor and let submit_package report the loc.
        with self.assertRaises(ValidationError) as cm:
            self.pool.submit_package([
                make_tx("ok", 1, 1, inputs=("z1:0",)),
                Tx("bad", 0, 1, ("z2:0",)),  # weight <= 0
            ])
        self.assertEqual(cm.exception.loc, "package[1].weight")
        self.assertEqual(_pool_snapshot(self.pool), before)

    def test_early_valid_tx_not_committed_when_later_tx_cycles(self):
        before = _pool_snapshot(self.pool)
        good = make_tx("child", 5, 2, inputs=("P:0",))
        # Cyclic pair among the later entries.
        a = make_tx("a", 5, 1, inputs=("b:0",))
        b = make_tx("b", 5, 1, inputs=("q:0",), depends=("a",))
        with self.assertRaises(ValidationError):
            self.pool.submit_package([good, a, b])
        self.assertEqual(_pool_snapshot(self.pool), before)
        self.assertNotIn("child", self.pool)
        self.assertIsNone(self.pool.spender_of("P:0"))

    def test_pool_remains_usable_after_rejected_package(self):
        with self.assertRaises(ValidationError):
            self.pool.submit_package([
                make_tx("a", 1, 1, inputs=("s:0",)),
                make_tx("b", 1, 1, inputs=("s:0",)),
            ])
        # Same outpoint double-spend rejected, but a well-formed retry
        # of the first tx alone succeeds.
        self.pool.submit_package(
            [make_tx("a", 1, 1, inputs=("s:0",))]
        )
        self.assertEqual(self.pool.spender_of("s:0"), "a")


class TestPackageOrderingInvariant(unittest.TestCase):
    def test_reported_order_is_topological(self):
        pool = AncestorPool()
        pool.add(make_tx("root", 1, 1, inputs=("r:0",)))
        # Package built over the pool ancestor; edges re-derived by hand.
        txs = [
            make_tx("tip", 1, 1, inputs=("left:0", "right:0",)),
            make_tx("right", 1, 1, inputs=("root:1",)),
            make_tx("left", 1, 1, inputs=("root:0",)),
        ]
        result = pool.submit_package(txs)
        universe = set(pool.ids())
        pos = {tid: i for i, tid in enumerate(result.order)}
        for tid in result.order:
            for par in _raw_parents(pool.get(tid), universe):
                if par in pos:  # only check within-package order
                    self.assertLess(pos[par], pos[tid])
        # Lex-smallest: root not in package order; among {left,right},
        # left precedes right, and tip is last.
        self.assertEqual(result.order, ("left", "right", "tip"))


class TestPackageSelectionCrossCheck(unittest.TestCase):
    """A pool built through submit_package must still select correctly.

    The independent exhaustive reference re-derives edges from raw Tx
    fields and never calls the core, so agreement is a genuine check.
    """

    def test_out_of_order_package_matches_brute_force(self):
        from tests._reference import brute_force_optimal

        base = [make_tx("b0", 3, 2, inputs=("base:0",))]
        # Package DAG over b0, deliberately scrambled on submission.
        pkg = [
            make_tx("tip", 4, 30, inputs=("m:0", "n:0",)),
            make_tx("m", 3, 4, inputs=("c:1",)),
            make_tx("c", 2, 3, inputs=("b0:0",)),
            make_tx("n", 3, 5, inputs=("c:0",)),
            make_tx("solo", 5, 9, inputs=("own:0",)),
        ]
        pool = AncestorPool()
        pool.add_many(base)
        pool.submit_package(pkg)

        all_txs = base + pkg
        for cap in (6, 8, 11, 14, 100):
            with self.subTest(cap=cap):
                r = select_package(pool, cap, node_budget=100_000)
                self.assertEqual(r.status, STATUS_PROVEN_OPTIMAL)
                fee, ids, weight = brute_force_optimal(all_txs, cap)
                self.assertEqual(r.total_fee, fee)
                self.assertEqual(set(r.selected), set(ids))
                self.assertEqual(r.total_weight, weight)


if __name__ == "__main__":
    unittest.main()
