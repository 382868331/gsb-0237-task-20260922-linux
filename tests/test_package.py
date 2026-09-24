"""Tests for the atomic ancestor-package submission ``add_package``.

Expected values are hand-derived (parent sets, topological orders, fee
and weight totals). Order-independence is checked by small exhaustive
permutation enumeration over fresh pools; the expected order is written
down, never produced by the implementation under test.
"""

import itertools
import unittest

from ancestor_pool import (
    AncestorPool,
    MAX_PACKAGE_SIZE,
    Tx,
    make_tx,
    ValidationError,
    MissingDependencyError,
    select_package,
    STATUS_PROVEN_OPTIMAL,
)


def _snapshot(pool):
    """Implementation-independent, hashable structural snapshot of a pool."""
    txs, parents = pool.edge_maps()
    return (
        tuple(sorted(txs)),
        tuple(sorted((tid, frozenset(ps)) for tid, ps in parents.items())),
        tuple(
            sorted(
                (tid, frozenset(pool.ancestors(tid))) for tid in sorted(txs)
            )
        ),
        tuple(
            sorted(
                (op, pool.spender_of(op))
                for txid in sorted(txs)
                for op in pool.get(txid).inputs
            )
        ),
    )


class TestPackageHappyPath(unittest.TestCase):
    def test_single_transaction_package(self):
        p = AncestorPool()
        r = p.add_package([make_tx("a", 7, 3, inputs=("c:0",))])
        self.assertEqual(len(r), 1)
        self.assertEqual(r.order, ("a",))
        self.assertEqual(r.ids, frozenset({"a"}))
        self.assertEqual(r.added_fee, 3)
        self.assertEqual(r.added_weight, 7)
        self.assertIn("a", p)

    def test_order_is_irrelevant_parents_any_position(self):
        # Hand-built DAG: root -> left,right ; {left,right} -> tip ;
        # plus an external-coin chain a -> b. All five txs are handed in
        # an order that is NOT topological (tip first, root not present).
        p = AncestorPool()
        package = [
            make_tx("tip", 5, 20, inputs=("left:0", "right:1")),
            make_tx("b", 4, 3, inputs=("a:0",)),
            make_tx("right", 5, 8, inputs=("root:1",)),
            make_tx("a", 4, 3, inputs=("extA:0",)),
            make_tx("left", 5, 8, inputs=("root:0",)),
        ]
        r = p.add_package(package)
        # Lexicographically-smallest topo order of the package itself.
        self.assertEqual(r.order, ("a", "b", "left", "right", "tip"))
        self.assertEqual(r.added_fee, 42)  # 20+3+8+3+8
        self.assertEqual(r.added_weight, 23)  # 5+4+5+4+5
        # Hand-derived parent sets.
        self.assertEqual(set(p.parents("tip")), {"left", "right"})
        self.assertEqual(set(p.parents("left")), set())
        self.assertEqual(set(p.parents("right")), set())
        self.assertEqual(set(p.parents("b")), {"a"})
        self.assertEqual(set(p.children("a")), {"b"})
        # Closures.
        self.assertEqual(set(p.ancestors("tip")), {"left", "right"})
        self.assertEqual(set(p.descendants("a")), {"b"})

    def test_existing_pool_ancestors_are_allowed(self):
        p = AncestorPool()
        p.add(make_tx("root", 10, 5, inputs=("coin:0",)))
        # The package child spends an output of an in-pool ancestor and
        # also names it explicitly via depends.
        r = p.add_package([
            make_tx("c", 6, 9, inputs=("root:0",), depends=("root",)),
        ])
        self.assertEqual(r.order, ("c",))
        self.assertEqual(set(p.parents("c")), {"root"})

    def test_eight_is_the_inclusive_size_boundary(self):
        p = AncestorPool()
        # 4 chains of length 2 = 8 txs.
        package = []
        for k in range(4):
            package.append(make_tx(f"p{k}", 1, 1, inputs=(f"c{k}:0",)))
        for k in range(4):
            package.append(make_tx(f"q{k}", 1, 1, inputs=(f"p{k}:0",)))
        r = p.add_package(package)
        self.assertEqual(len(r), 8)
        self.assertEqual(len(p), 8)
        self.assertEqual(MAX_PACKAGE_SIZE, 8)

    def test_accepts_generator_and_reports_topological_order(self):
        p = AncestorPool()
        p.add(make_tx("r", 1, 1, inputs=("cr:0",)))

        def gen():
            # reversed: child before parent; pool parent r at the bottom.
            yield make_tx("z", 1, 1, inputs=("y:0",))
            yield make_tx("y", 1, 1, inputs=("r:0",))

        res = p.add_package(gen())
        self.assertEqual(res.order, ("y", "z"))
        self.assertEqual(set(p.parents("z")), {"y"})
        self.assertEqual(set(p.parents("y")), {"r"})

    def test_second_package_builds_on_first(self):
        p = AncestorPool()
        p.add_package([make_tx("a", 2, 4, inputs=("ca:0",))])
        p.add_package([
            make_tx("b", 2, 4, inputs=("a:0",)),
            make_tx("c", 2, 4, inputs=("a:1",)),
        ])
        self.assertEqual(set(p.ids()), {"a", "b", "c"})
        self.assertEqual(set(p.children("a")), {"b", "c"})

    def test_package_then_select_sees_the_whole_dag(self):
        p = AncestorPool()
        p.add_package([
            make_tx("p", 10, 1, inputs=("cp:0",)),
            make_tx("h", 10, 9, inputs=("p:0",)),
        ])
        r = select_package(p, 25, node_budget=10_000)
        self.assertEqual(r.status, STATUS_PROVEN_OPTIMAL)
        self.assertEqual(list(r.selected), ["p", "h"])
        self.assertEqual(r.total_fee, 10)


class TestOrderIndependenceEnumeration(unittest.TestCase):
    def test_all_permutations_are_equivalent(self):
        # Package: a(ext) -> b ; root(in pool) -> s -> t. Every one of
        # the 4! = 24 submission orders must yield an identical pool and
        # the same hand-written topo order.
        package = [
            make_tx("a", 3, 2, inputs=("ea:0",)),
            make_tx("b", 3, 2, inputs=("a:0",)),
            make_tx("s", 3, 2, inputs=("root:0",)),
            make_tx("t", 3, 2, inputs=("s:0",)),
        ]
        expected_parents = {
            "a": frozenset(),
            "b": frozenset({"a"}),
            "s": frozenset({"root"}),
            "t": frozenset({"s"}),
        }
        snaps = set()
        for perm in itertools.permutations(package):
            p = AncestorPool()
            p.add(make_tx("root", 3, 1, inputs=("er:0",)))
            res = p.add_package(list(perm))
            self.assertEqual(res.order, ("a", "b", "s", "t"))
            got = {tid: p.parents(tid) for tid in expected_parents}
            self.assertEqual(got, expected_parents)
            snaps.add(_snapshot(p))
        self.assertEqual(len(snaps), 1)  # every permutation identical


class TestPackageBoundaries(unittest.TestCase):
    def test_nine_transactions_rejected(self):
        p = AncestorPool()
        with self.assertRaises(ValidationError) as cm:
            p.add_package(
                [make_tx(f"m{i}", 1, 0, inputs=(f"em{i}:0",)) for i in range(9)]
            )
        self.assertEqual(cm.exception.kind, "package_too_large")
        self.assertEqual(len(p), 0)

    def test_empty_package_rejected(self):
        p = AncestorPool()
        with self.assertRaises(ValidationError) as cm:
            p.add_package([])
        self.assertEqual(cm.exception.kind, "empty_package")

    def test_custom_max_size(self):
        p = AncestorPool()
        four = [make_tx(f"m{i}", 1, 0, inputs=(f"em{i}:0",)) for i in range(4)]
        with self.assertRaises(ValidationError) as cm:
            p.add_package(four, max_size=3)
        self.assertEqual(cm.exception.kind, "package_too_large")
        r = p.add_package(four, max_size=4)
        self.assertEqual(len(r), 4)

    def test_max_size_must_be_valid_int(self):
        p = AncestorPool()
        one = [make_tx("m", 1, 0, inputs=("em:0",))]
        for bad in (0, 9, True, 1.5, "3"):
            with self.assertRaises(ValidationError):
                p.add_package(one, max_size=bad)
        self.assertEqual(len(p), 0)

    def test_pool_capacity_is_checked_atomically(self):
        p = AncestorPool()
        for i in range(25):
            p.add(make_tx(f"t{i:02d}", 1, 0, inputs=(f"e{i}:0",)))
        # 25 + 8 = 33 > 30: rejected wholesale.
        with self.assertRaises(ValidationError) as cm:
            p.add_package(
                [make_tx(f"n{i}", 1, 0, inputs=(f"en{i}:0",)) for i in range(8)]
            )
        self.assertEqual(cm.exception.kind, "capacity")
        self.assertEqual(len(p), 25)
        # 25 + 5 = 30 exactly: accepted.
        res = p.add_package(
            [make_tx(f"k{i}", 1, 0, inputs=(f"ek{i}:0",)) for i in range(5)]
        )
        self.assertEqual(len(res), 5)
        self.assertEqual(len(p), 30)

    def test_non_iterable_input_rejected(self):
        p = AncestorPool()
        for bad in (None, 42, object()):
            with self.assertRaises(ValidationError):
                p.add_package(bad)
        # A bare str must not be treated as an iterable of Tx.
        with self.assertRaises(ValidationError):
            p.add_package("abc")
        self.assertEqual(len(p), 0)


class TestPackageFailuresAtomic(unittest.TestCase):
    def setUp(self):
        self.pool = AncestorPool()
        self.pool.add(make_tx("root", 10, 5, inputs=("coin:0",)))
        self.pool.add(make_tx("child", 10, 7, inputs=("root:0",)))
        self.before = _snapshot(self.pool)

    def _assert_untouched(self):
        self.assertEqual(_snapshot(self.pool), self.before)

    def test_cycle_via_depends_rejected(self):
        x = make_tx("x", 1, 1, inputs=("cx:0",), depends=("y",))
        y = make_tx("y", 1, 1, inputs=("x:0",))
        with self.assertRaises(ValidationError) as cm:
            self.pool.add_package([x, y])
        self.assertEqual(cm.exception.kind, "cycle")
        self._assert_untouched()

    def test_cycle_via_outpoints_rejected(self):
        # d spends c:0 and c spends d:0 -> edge cycle, no depends used.
        c = make_tx("c", 1, 1, inputs=("d:0",))
        d = make_tx("d", 1, 1, inputs=("c:0",))
        with self.assertRaises(ValidationError) as cm:
            self.pool.add_package([c, d])
        self.assertEqual(cm.exception.kind, "cycle")
        self._assert_untouched()

    def test_self_dependency_rejected(self):
        with self.assertRaises(ValidationError):
            self.pool.add_package(
                [make_tx("s", 1, 1, inputs=("es:0",), depends=("s",))]
            )
        self._assert_untouched()

    def test_intra_package_double_spend_rejected(self):
        with self.assertRaises(ValidationError) as cm:
            self.pool.add_package([
                make_tx("u", 1, 1, inputs=("z:0",)),
                make_tx("v", 1, 1, inputs=("z:0",)),
            ])
        self.assertEqual(cm.exception.kind, "double_spend")
        self._assert_untouched()

    def test_spending_pool_owned_outpoint_is_append_only(self):
        # coin:0 is already spent by root in the pool.
        with self.assertRaises(ValidationError) as cm:
            self.pool.add_package(
                [make_tx("rbf", 1, 100, inputs=("coin:0",))]
            )
        self.assertEqual(cm.exception.kind, "conflict_with_pool")
        self._assert_untouched()
        self.assertEqual(self.pool.spender_of("coin:0"), "root")

    def test_duplicate_id_within_package(self):
        with self.assertRaises(ValidationError) as cm:
            self.pool.add_package([
                make_tx("dup", 1, 1, inputs=("e1:0",)),
                make_tx("dup", 1, 1, inputs=("e2:0",)),
            ])
        self.assertEqual(cm.exception.kind, "duplicate")
        self._assert_untouched()

    def test_duplicate_id_against_pool(self):
        with self.assertRaises(ValidationError) as cm:
            self.pool.add_package(
                [make_tx("root", 1, 1, inputs=("ex:0",))]
            )
        self.assertEqual(cm.exception.kind, "duplicate")
        self._assert_untouched()

    def test_missing_parent_locatable(self):
        with self.assertRaises(MissingDependencyError) as cm:
            self.pool.add_package([
                make_tx("orphan", 1, 1, inputs=("eo:0",), depends=("ghost",))
            ])
        self.assertEqual(cm.exception.missing, ("ghost",))
        self.assertEqual(cm.exception.loc, "txs[0].depends[0]")
        self._assert_untouched()

    def test_missing_creator_outpoint_is_external_coin_not_error(self):
        # Unknown creator prefix: external coin, no parent required.
        r = self.pool.add_package(
            [make_tx("ext", 1, 1, inputs=("nobody:9",))]
        )
        self.assertEqual(r.order, ("ext",))
        self.assertEqual(set(self.pool.parents("ext")), set())

    def test_invalid_field_has_indexed_loc(self):
        # Raw Tx (not make_tx) so the invalid field survives until the
        # package validator reports its indexed location.
        bad_weight = Tx("b", 0, 1, inputs=("eb:0",))
        with self.assertRaises(ValidationError) as cm:
            self.pool.add_package([
                make_tx("g0", 1, 1, inputs=("e0:0",)),
                make_tx("g1", 1, 1, inputs=("e1:0",)),
                bad_weight,
            ])
        self.assertEqual(cm.exception.loc, "txs[2].weight")
        self._assert_untouched()

    def test_valid_package_still_accepted_after_failure(self):
        with self.assertRaises(ValidationError):
            self.pool.add_package([
                make_tx("u", 1, 1, inputs=("z:0",)),
                make_tx("v", 1, 1, inputs=("z:0",)),
            ])
        r = self.pool.add_package([
            make_tx("w", 2, 6, inputs=("ew:0",)),
            make_tx("x", 2, 6, inputs=("w:0",), depends=("root",)),
        ])
        self.assertEqual(r.order, ("w", "x"))
        self.assertEqual(set(self.pool.ids()), {"root", "child", "w", "x"})
        self.assertEqual(set(self.pool.parents("x")), {"w", "root"})

    def test_failure_mid_enumeration_never_touches_live_pool(self):
        # A package that passes field checks but fails on a cycle: even
        # though several txs would have been inserted, not even closures
        # of pre-existing txs may change.
        before_desc = {
            tid: self.pool.descendants(tid) for tid in self.pool.ids()
        }
        with self.assertRaises(ValidationError):
            self.pool.add_package([
                make_tx("a2", 1, 1, inputs=("ca2:0",)),
                make_tx("b2", 1, 1, inputs=("a2:0",), depends=("c2",)),
                make_tx("c2", 1, 1, inputs=("b2:0",)),
            ])
        self._assert_untouched()
        for tid, desc in before_desc.items():
            self.assertEqual(self.pool.descendants(tid), desc)


if __name__ == "__main__":
    unittest.main()
