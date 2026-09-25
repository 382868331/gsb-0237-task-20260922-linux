"""Tests for AncestorPool.confirm (block confirmation semantics)."""

import unittest

from ancestor_pool import (
    AncestorPool,
    BlockTx,
    DuplicateTxIdError,
    MissingDependencyError,
    ValidationError,
    make_tx,
    select_package,
)

from tests._reference import brute_force_optimal


def pool_with(*txs):
    pool = AncestorPool()
    pool.add_many(list(txs))
    return pool


class TestParentConfirmedChildKept(unittest.TestCase):
    def test_child_survives_parent_confirmation(self):
        parent = make_tx("p", 10, 5, inputs=("ext:0",))
        child = make_tx("c", 10, 9, inputs=("p:0", "ext:1"), depends=("p",))
        grand = make_tx("g", 10, 1, inputs=("c:0",), depends=("c",))
        pool = pool_with(parent, child, grand)

        result = pool.confirm([BlockTx(parent, created=("p:0",))])

        self.assertEqual(result.pool_confirmed_ids, frozenset({"p"}))
        self.assertEqual(result.conflict_removed_ids, frozenset())
        self.assertNotIn("p", pool)
        self.assertIn("c", pool)
        self.assertIn("g", pool)
        # The child's dependency is now a confirmed output, not a pool tx.
        self.assertEqual(pool.parents("c"), frozenset())
        self.assertEqual(pool.ancestors("g"), frozenset({"c"}))
        self.assertEqual(pool.confirmed_outpoints(), frozenset({"p:0"}))
        self.assertEqual(pool.confirmed_spent(), frozenset({"ext:0"}))
        self.assertTrue(pool.is_confirmed("p"))

    def test_select_after_confirm_excludes_confirmed_weight_and_fee(self):
        parent = make_tx("p", 10, 100, inputs=("ext:0",))
        child = make_tx("c", 10, 9, inputs=("p:0",), depends=("p",))
        pool = pool_with(parent, child)
        pool.confirm([BlockTx(parent, created=("p:0",))])
        # Only the child remains; the confirmed parent contributes
        # neither weight nor fee and is not required for closure.
        result = select_package(pool, 10)
        self.assertEqual(list(result.selected), ["c"])
        self.assertEqual(result.total_fee, 9)
        self.assertEqual(result.total_weight, 10)

    def test_add_after_confirm_rules(self):
        parent = make_tx("p", 10, 5, inputs=("ext:0",))
        pool = pool_with(parent)
        pool.confirm([BlockTx(parent, created=("p:0",))])
        # Spending a confirmed-created output is fine (external coin).
        pool.add(make_tx("n", 5, 5, inputs=("p:0",)))
        self.assertIn("n", pool)
        # Re-adding a confirmed id is rejected.
        with self.assertRaises(DuplicateTxIdError):
            pool.add(make_tx("p", 1, 1, inputs=("z:0",)))
        # Spending an already confirmed-spent outpoint is rejected.
        with self.assertRaises(ValidationError):
            pool.add(make_tx("m", 1, 1, inputs=("ext:0",)))


class TestBlockConflictRemovesDescendants(unittest.TestCase):
    def test_conflict_evicts_descendants_not_confirmed(self):
        # Pool chain a -> b -> c; block tx spends a's input.
        a = make_tx("a", 10, 5, inputs=("ext:0",))
        b = make_tx("b", 10, 5, inputs=("a:0",), depends=("a",))
        c = make_tx("c", 10, 5, inputs=("b:0",), depends=("b",))
        keep = make_tx("k", 10, 5, inputs=("ext:9",))
        pool = pool_with(a, b, c, keep)

        block_tx = make_tx("w", 10, 5, inputs=("ext:0",))
        result = pool.confirm([BlockTx(block_tx, created=("w:0",))])

        self.assertEqual(result.conflict_removed_ids, frozenset({"a", "b", "c"}))
        self.assertEqual(result.pool_confirmed_ids, frozenset())
        self.assertEqual(pool.ids(), frozenset({"k"}))

    def test_confirmed_item_reported_once(self):
        # One block confirms pool tx "x" and conflicts with pool tx "y":
        # "x" is reported as confirmed only, never as conflict-removed.
        x = make_tx("x", 10, 5, inputs=("ext:0",))
        y = make_tx("y", 10, 5, inputs=("ext:1",))
        pool = pool_with(x, y)
        w = make_tx("w", 10, 5, inputs=("ext:1",))  # conflicts with y
        result = pool.confirm(
            [BlockTx(x, created=("x:0",)), BlockTx(w, created=("w:0",))]
        )
        self.assertEqual(result.pool_confirmed_ids, frozenset({"x"}))
        self.assertEqual(result.conflict_removed_ids, frozenset({"y"}))
        self.assertNotIn("x", result.conflict_removed_ids)
        self.assertEqual(pool.ids(), frozenset())


class TestBlockConfirmsParentAndChild(unittest.TestCase):
    def test_parent_child_one_block_grandchild_kept(self):
        p = make_tx("p", 10, 5, inputs=("ext:0",))
        c = make_tx("c", 10, 5, inputs=("p:0",), depends=("p",))
        g = make_tx("g", 10, 7, inputs=("c:0",), depends=("c",))
        pool = pool_with(p, c, g)

        result = pool.confirm(
            [BlockTx(p, created=("p:0",)), BlockTx(c, created=("c:0",))]
        )

        self.assertEqual(result.confirmed_ids, ("p", "c"))
        self.assertEqual(result.pool_confirmed_ids, frozenset({"p", "c"}))
        self.assertEqual(result.conflict_removed_ids, frozenset())
        self.assertEqual(pool.ids(), frozenset({"g"}))
        self.assertEqual(pool.parents("g"), frozenset())
        # p:0 was created and spent within the same block: only c:0
        # remains an unspent confirmed outpoint.
        self.assertEqual(pool.confirmed_outpoints(), frozenset({"c:0"}))
        self.assertIn("p:0", pool.confirmed_spent())


class TestInvalidBlockRollsBack(unittest.TestCase):
    def setUp(self):
        self.a = make_tx("a", 10, 5, inputs=("ext:0",))
        self.b = make_tx("b", 10, 5, inputs=("a:0",), depends=("a",))
        self.pool = pool_with(self.a, self.b)
        self.before = (
            dict(self.pool._txs),
            dict(self.pool._outpoint_owner),
            set(self.pool._confirmed_ids),
        )

    def _assert_unchanged(self):
        self.assertEqual(self.pool._txs, self.before[0])
        self.assertEqual(self.pool._outpoint_owner, self.before[1])
        self.assertEqual(self.pool._confirmed_ids, self.before[2])

    def test_duplicate_id_in_block(self):
        t = make_tx("w", 1, 1, inputs=("e:0",))
        with self.assertRaises(DuplicateTxIdError):
            self.pool.confirm([BlockTx(t), BlockTx(t)])
        self._assert_unchanged()

    def test_intra_block_double_spend(self):
        t1 = make_tx("w1", 1, 1, inputs=("e:0",))
        t2 = make_tx("w2", 1, 1, inputs=("e:0",))
        with self.assertRaises(ValidationError):
            self.pool.confirm([BlockTx(t1), BlockTx(t2)])
        self._assert_unchanged()

    def test_bad_dependency_order(self):
        # Child confirmed before its parent: order violation.
        child = make_tx("wc", 1, 1, inputs=("wp:0",), depends=("wp",))
        parent = make_tx("wp", 1, 1, inputs=("e:0",))
        with self.assertRaises(MissingDependencyError):
            self.pool.confirm([BlockTx(child), BlockTx(parent, ("wp:0",))])
        self._assert_unchanged()

    def test_dependency_on_unconfirmed_pool_tx_rejected(self):
        t = make_tx("w", 1, 1, inputs=("a:0",), depends=("a",))
        with self.assertRaises(MissingDependencyError):
            self.pool.confirm([BlockTx(t)])
        self._assert_unchanged()

    def test_reconfirming_rejected(self):
        t = make_tx("w", 1, 1, inputs=("e:0",))
        self.pool.confirm([BlockTx(t, ("w:0",))])
        with self.assertRaises(DuplicateTxIdError):
            self.pool.confirm([BlockTx(t)])
        self.assertEqual(self.pool.ids(), frozenset({"a", "b"}))

    def test_spending_confirmed_spent_outpoint_rejected(self):
        t1 = make_tx("w1", 1, 1, inputs=("e:0",))
        self.pool.confirm([BlockTx(t1, ("w1:0",))])
        t2 = make_tx("w2", 1, 1, inputs=("e:0",))
        with self.assertRaises(ValidationError):
            self.pool.confirm([BlockTx(t2)])


class TestConfirmThenSelectCrossCheck(unittest.TestCase):
    """Small-pool enumeration cross-check after confirmations."""

    def test_select_matches_brute_force_after_confirm(self):
        from tests._data import gen_txs

        txs = gen_txs(seed=7, n=10)
        pool = pool_with(*txs[:10])
        # Confirm the first three (they form a valid prefix: parents
        # always precede children in the generator).
        block = [
            BlockTx(t, created=tuple(f"{t.txid}:{i}" for i in range(1, 11)))
            for t in txs[:3]
        ]
        pool.confirm(block)
        # The reference derives parents from raw Tx fields; dependencies
        # on confirmed transactions are satisfied by the ledger, so they
        # are stripped for the cross-check (inputs are ignored by the
        # reference when the creator is absent).
        remaining = []
        for tid in pool.ids():
            tx = pool.get(tid)
            remaining.append(
                make_tx(
                    tx.txid,
                    tx.weight,
                    tx.fee,
                    inputs=tx.inputs,
                    depends=tuple(d for d in tx.depends if d in pool),
                )
            )
        for budget in (15, 25, 40):
            result = select_package(pool, budget)
            fee, ids, weight = brute_force_optimal(remaining, budget)
            self.assertTrue(result.proven_optimal)
            self.assertEqual(result.total_fee, fee)
            self.assertEqual(tuple(sorted(result.selected)), tuple(ids))
            self.assertEqual(result.total_weight, weight)


if __name__ == "__main__":
    unittest.main()
