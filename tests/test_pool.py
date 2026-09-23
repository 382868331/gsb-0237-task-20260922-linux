import math
import unittest

from ancestor_pool import (
    AncestorPool,
    make_tx,
    ValidationError,
    DuplicateTxIdError,
    MissingDependencyError,
    ReplacementRejectedError,
    REASON_FEE_NOT_GREATER,
    REASON_FEE_RATE_NOT_HIGHER,
    REASON_DEPENDS_ON_REMOVED,
)


class TestValidation(unittest.TestCase):
    def test_rejects_bool_for_int_fields(self):
        with self.assertRaises(ValidationError) as cm:
            make_tx("a", True, 1)
        self.assertEqual(cm.exception.loc, "tx.weight")
        with self.assertRaises(ValidationError) as cm:
            make_tx("a", 10, False)
        self.assertEqual(cm.exception.loc, "tx.fee")

    def test_rejects_float_so_nan_inf_cannot_enter(self):
        # The numeric model is int-only; NaN/Infinity cannot ever enter.
        for bad in (math.nan, math.inf, -math.inf, 1.5, 0.0):
            with self.assertRaises(ValidationError):
                make_tx("a", bad, 1)
            with self.assertRaises(ValidationError):
                make_tx("a", 10, bad)

    def test_rejects_bad_strings(self):
        with self.assertRaises(ValidationError):
            make_tx("", 10, 1)
        with self.assertRaises(ValidationError):
            make_tx("a\x00", 10, 1)
        with self.assertRaises(ValidationError):
            make_tx(3, 10, 1)

    def test_rejects_bad_numbers(self):
        with self.assertRaises(ValidationError):
            make_tx("a", 0, 1)
        with self.assertRaises(ValidationError):
            make_tx("a", -3, 1)
        with self.assertRaises(ValidationError):
            make_tx("a", 10, -1)

    def test_rejects_duplicate_inputs(self):
        with self.assertRaises(ValidationError):
            make_tx("a", 10, 1, inputs=("x:0", "x:0"))

    def test_self_dependency_rejected(self):
        with self.assertRaises(ValidationError):
            make_tx("a", 10, 1, depends=("a",))
        with self.assertRaises(ValidationError):
            make_tx("a", 10, 1, inputs=("a:0",))

    def test_error_loc_is_specific(self):
        with self.assertRaises(ValidationError) as cm:
            make_tx("a", "heavy", 1)
        self.assertIn("weight", cm.exception.loc)

    def test_external_coin_is_valid(self):
        tx = make_tx("a", 10, 1, inputs=("1JimmyHendrix:7",))
        self.assertEqual(tx.inputs, ("1JimmyHendrix:7",))


class TestBasicPool(unittest.TestCase):
    def test_insert_and_outpoint_edge(self):
        p = AncestorPool()
        p.add(make_tx("p", 10, 4, inputs=("coin:0",)))
        r = p.add(make_tx("c", 20, 9, inputs=("p:0",)))
        self.assertFalse(r.replaced)
        self.assertEqual(set(p.ids()), {"p", "c"})
        self.assertEqual(set(p.parents("c")), {"p"})
        self.assertEqual(set(p.children("p")), {"c"})

    def test_missing_explicit_dependency_is_atomic(self):
        p = AncestorPool()
        with self.assertRaises(MissingDependencyError) as cm:
            p.add(make_tx("c", 10, 1, depends=("ghost",)))
        self.assertIn("ghost", cm.exception.missing)
        self.assertEqual(len(p), 0)

    def test_duplicate_id(self):
        p = AncestorPool()
        p.add(make_tx("a", 10, 1, inputs=("x:0",)))
        with self.assertRaises(DuplicateTxIdError):
            p.add(make_tx("a", 10, 2, inputs=("y:0",)))

    def test_capacity_boundary(self):
        p = AncestorPool()
        for i in range(30):
            p.add(make_tx(f"t{i}", 1, 0, inputs=(f"e{i}:0",)))
        self.assertEqual(len(p), 30)
        with self.assertRaises(ValidationError):
            p.add(make_tx("overflow", 1, 0, inputs=("e99:0",)))
        # A conflicting add at capacity is a replacement, not growth.
        res = p.add(make_tx("r0", 2, 1, inputs=("e0:0",)))
        self.assertTrue(res.replaced)
        self.assertEqual(len(p), 30)

    def test_external_coin_needs_no_pool_entry(self):
        p = AncestorPool()
        p.add(make_tx("a", 10, 1, inputs=("some:7",)))
        self.assertEqual(len(p), 1)


class TestReplacement(unittest.TestCase):
    def _chain(self, fees, weights=10):
        # p spends coin:0 ; c spends p:0 ; g spends c:0.
        if isinstance(weights, int):
            weights = [weights] * len(fees)
        p = AncestorPool()
        p.add(make_tx("p", weights[0], fees[0], inputs=("coin:0",)))
        if len(fees) > 1:
            p.add(make_tx("c", weights[1], fees[1], inputs=("p:0",)))
        if len(fees) > 2:
            p.add(make_tx("g", weights[2], fees[2], inputs=("c:0",)))
        return p

    def test_replace_conflict_and_all_descendants(self):
        pool = self._chain([10, 9, 9])
        res = pool.add(make_tx("r", 30, 100, inputs=("coin:0",)))
        self.assertEqual(set(res.removed_ids), {"p", "c", "g"})
        self.assertEqual(res.removed_fee, 28)
        self.assertEqual(set(pool.ids()), {"r"})

    def test_fee_must_be_strictly_greater(self):
        pool = self._chain([10, 5])
        before = pool.ids()
        with self.assertRaises(ReplacementRejectedError) as cm:
            pool.add(make_tx("r", 10, 15, inputs=("coin:0",)))
        self.assertEqual(cm.exception.reason, REASON_FEE_NOT_GREATER)
        self.assertEqual(pool.ids(), before)

    def test_fee_rate_must_be_strictly_higher(self):
        pool = self._chain([100], weights=[100])
        with self.assertRaises(ReplacementRejectedError) as cm:
            pool.add(make_tx("r", 200, 101, inputs=("coin:0",)))
        self.assertEqual(cm.exception.reason, REASON_FEE_RATE_NOT_HIGHER)
        self.assertIn("p", pool.ids())

    def test_equal_fee_rate_is_rejected(self):
        pool = self._chain([10], weights=[10])
        with self.assertRaises(ReplacementRejectedError) as cm:
            pool.add(make_tx("r", 20, 20, inputs=("coin:0",)))
        self.assertEqual(cm.exception.reason, REASON_FEE_RATE_NOT_HIGHER)

    def test_incoming_must_not_depend_on_removed(self):
        # Low-fee parent, high-fee child: the replacement must not be
        # allowed to lean on any evicted transaction.
        pool = self._chain([1, 100])
        with self.assertRaises(ReplacementRejectedError) as cm:
            pool.add(
                make_tx("r", 10, 500, inputs=("coin:0",), depends=("c",))
            )
        self.assertEqual(cm.exception.reason, REASON_DEPENDS_ON_REMOVED)
        self.assertEqual(set(pool.ids()), {"p", "c"})

    def test_orphan_after_replacement_is_removed(self):
        # Low-fee parent -> high-fee child -> grandchild. Replacing the
        # parent evicts the high-fee child (orphan) as well.
        pool = self._chain([1, 1000, 1000])
        with self.assertRaises(ReplacementRejectedError):
            pool.add(make_tx("r", 10, 2001, inputs=("coin:0",)))  # equal fee
        res = pool.add(make_tx("r", 1, 2002, inputs=("coin:0",)))
        self.assertEqual(set(res.removed_ids), {"p", "c", "g"})
        self.assertNotIn("c", pool.ids())

    def test_indirect_conflict_through_descendant_outpoint(self):
        # A spends coin:0; B spends A:0; C spends B:0. R spends A:0, so
        # it directly conflicts only with B; B's descendant C must be
        # evicted too, while the ancestor A survives and becomes R's
        # parent.
        p = AncestorPool()
        p.add(make_tx("A", 10, 5, inputs=("coin:0",)))
        p.add(make_tx("B", 10, 5, inputs=("A:0",)))
        p.add(make_tx("C", 10, 5, inputs=("B:0",)))
        res = p.add(make_tx("R", 10, 40, inputs=("A:0",)))
        self.assertEqual(set(res.removed_ids), {"B", "C"})
        self.assertIn("A", p.ids())
        self.assertEqual(set(p.parents("R")), {"A"})

    def test_replacement_failure_is_atomic(self):
        pool = self._chain([10, 10])
        with self.assertRaises(ReplacementRejectedError):
            pool.add(make_tx("r", 100, 21, inputs=("coin:0",)))
        txs, parents = pool.edge_maps()
        self.assertEqual(set(txs), {"p", "c"})
        self.assertEqual(parents["c"], {"p"})
        self.assertEqual(pool.spender_of("coin:0"), "p")


class TestBatch(unittest.TestCase):
    def test_batch_double_spend_rejected_wholesale(self):
        p = AncestorPool()
        a = make_tx("a", 10, 1, inputs=("x:0",))
        b = make_tx("b", 10, 1, inputs=("x:0",))
        with self.assertRaises(ValidationError):
            p.add_many([a, b])
        self.assertEqual(len(p), 0)

    def test_batch_atomic_when_later_replacement_fails(self):
        p = AncestorPool()
        p.add(make_tx("p", 10, 10, inputs=("coin:0",)))
        good = make_tx("g", 10, 5, inputs=("p:0",))
        bad = make_tx("r", 100, 11, inputs=("coin:0",))
        with self.assertRaises(ReplacementRejectedError):
            p.add_many([good, bad])
        self.assertEqual(set(p.ids()), {"p"})

    def test_batch_success(self):
        p = AncestorPool()
        rs = p.add_many([
            make_tx("a", 10, 4, inputs=("c1:0",)),
            make_tx("b", 10, 4, inputs=("a:0",)),
        ])
        self.assertEqual(len(rs), 2)
        self.assertEqual(set(p.ids()), {"a", "b"})


if __name__ == "__main__":
    unittest.main()
