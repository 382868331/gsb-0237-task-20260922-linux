import unittest

from ancestor_pool import (
    AncestorPool,
    make_tx,
    validate_batch,
    select_package,
    ValidationError,
    MissingDependencyError,
)


class TestBatchValidation(unittest.TestCase):
    def test_duplicate_id_against_pool(self):
        p = AncestorPool()
        p.add(make_tx("a", 10, 1, inputs=("x:0",)))
        with self.assertRaises(ValidationError):
            p.add_many([make_tx("a", 10, 2, inputs=("y:0",))])
        self.assertEqual(set(p.ids()), {"a"})

    def test_batch_cycle_is_rejected(self):
        # a depends on b and b spends a:0 -> a cycle across the batch.
        a = make_tx("a", 10, 1, inputs=("x:0",), depends=("b",))
        b = make_tx("b", 10, 1, inputs=("a:0",))
        with self.assertRaises(ValidationError) as cm:
            validate_batch([a, b])
        self.assertEqual(cm.exception.kind, "cycle")

    def test_batch_cycle_against_existing_pool(self):
        p = AncestorPool()
        p.add(make_tx("root", 10, 1, inputs=("x:0",)))
        # c depends on d; d spends c:0 (c new) -> cycle entirely in batch
        c = make_tx("c", 10, 1, inputs=("root:0",), depends=("d",))
        d = make_tx("d", 10, 1, inputs=("c:0",))
        with self.assertRaises(ValidationError):
            p.add_many([c, d])
        self.assertEqual(set(p.ids()), {"root"})

    def test_batch_missing_dependency_locatable(self):
        with self.assertRaises(MissingDependencyError) as cm:
            validate_batch([make_tx("a", 10, 1, depends=("ghost",))])
        self.assertEqual(cm.exception.missing, ("ghost",))
        self.assertIn("depends", cm.exception.loc)

    def test_standalone_batch_double_spend(self):
        with self.assertRaises(ValidationError) as cm:
            validate_batch([
                make_tx("a", 10, 1, inputs=("o:0",)),
                make_tx("b", 10, 1, inputs=("o:0",)),
            ])
        self.assertEqual(cm.exception.kind, "double_spend")


class TestTopologicalOutput(unittest.TestCase):
    def test_diamond_graph_order(self):
        # root -> left,right -> tip ; lex-smallest topo order of the
        # full chosen set is root,left,right,tip.
        p = AncestorPool()
        p.add_many([
            make_tx("root", 1, 1, inputs=("c:0",)),
            make_tx("right", 1, 1, inputs=("root:1",)),
            make_tx("left", 1, 1, inputs=("root:0",)),
            make_tx("tip", 1, 1, inputs=("left:0", "right:0")),
        ])
        r = select_package(p, 100, node_budget=1000)
        self.assertEqual(
            list(r.selected), ["root", "left", "right", "tip"]
        )
        pos = {t: i for i, t in enumerate(r.selected)}
        self.assertLess(pos["root"], pos["left"])
        self.assertLess(pos["left"], pos["tip"])
        self.assertLess(pos["right"], pos["tip"])


class TestReferenceBound(unittest.TestCase):
    def test_reference_refuses_more_than_16(self):
        from tests._reference import brute_force_optimal
        txs = [make_tx(f"t{i}", 1, 0, inputs=(f"c{i}:0",)) for i in range(17)]
        with self.assertRaises(ValueError):
            brute_force_optimal(txs, 100)


if __name__ == "__main__":
    unittest.main()
