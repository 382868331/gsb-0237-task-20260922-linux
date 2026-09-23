"""Dependency-closed maximum-fee package selection under a weight budget.

Problem
-------
Choose a subset *S* of pool transactions maximising total fee subject to

* ``sum(weight) <= block_weight``;
* *S* is ancestor-closed (parents via explicit ``depends`` or spent
  creator outpoints);
* ties are broken by the lexicographically smallest tuple of selected
  ids, each tuple written in ascending id order and compared with
  standard sequence ordering (a proper prefix is smaller). This is the
  Python comparison ``tuple(sorted(ids_a)) < tuple(sorted(ids_b))`` and
  is the only tie-break implemented (documented in README).

Return the selection in topological (parent-first, id tie-broken) order
plus a reason for every excluded transaction.

Search and budget
-----------------
The problem is NP-hard in general, so the solver is an explicit
branch-and-bound DFS. Everything is integer: ratios are ordered once by
cross multiplication and the fractional-knapsack bound uses

    bound = cur_fee + sum(full fees) + (fee_j * cap) // weight_j

for the single split item. Flooring the fractional bound is safe
because every real completion has an integer fee, so pruning whenever
``bound < incumbent_fee`` can never discard an optimum. No float is used.

A *node budget* (expanded DFS nodes, one unit per expansion) bounds the
work and must always be supplied explicitly. The two terminal outcomes
are kept distinct:

* ``proven_optimal`` -- the search completed, so the returned package is
  the global optimum with the documented tie-break;
* ``budget_exhausted`` -- the node budget ran out first. The best
  feasible package found is returned without any optimality claim (its
  tie-break may be unsettled too), and every excluded transaction whose
  status could not be settled is reported ``budget_inconclusive`` rather
  than given a definite reason.
"""

from functools import cmp_to_key

from .errors import ValidationError
from .graph import closure_maps, topo_order
from .pool import AncestorPool

MAX_TRANSACTIONS = 30
DEFAULT_BUDGET = 200_000

# Exclusion reason codes (stable, public strings).
CODE_OVERWEIGHT = "overweight_even_alone"
CODE_ANCESTOR_EXCLUDED = "ancestor_excluded"
CODE_NOT_SELECTED = "not_selected"
CODE_BUDGET_INCONCLUSIVE = "budget_inconclusive"

STATUS_PROVEN_OPTIMAL = "proven_optimal"
STATUS_BUDGET_EXHAUSTED = "budget_exhausted"


class Exclusion:
    """Why one transaction was left out of the package."""

    __slots__ = ("txid", "code", "detail")

    def __init__(self, txid, code, detail):
        self.txid = txid
        self.code = code
        self.detail = detail

    def __repr__(self):
        return f"Exclusion(txid={self.txid!r}, code={self.code!r})"

    def __eq__(self, other):
        return (
            isinstance(other, Exclusion)
            and self.txid == other.txid
            and self.code == other.code
            and self.detail == other.detail
        )


class SelectionResult:
    """Result of :func:`select_package`."""

    __slots__ = (
        "selected",
        "excluded",
        "total_fee",
        "total_weight",
        "status",
        "nodes_used",
        "node_budget",
        "block_weight",
    )

    def __init__(
        self,
        selected,
        excluded,
        total_fee,
        total_weight,
        status,
        nodes_used,
        node_budget,
        block_weight,
    ):
        self.selected = tuple(selected)
        self.excluded = tuple(excluded)
        self.total_fee = total_fee
        self.total_weight = total_weight
        self.status = status
        self.nodes_used = nodes_used
        self.node_budget = node_budget
        self.block_weight = block_weight

    @property
    def proven_optimal(self):
        return self.status == STATUS_PROVEN_OPTIMAL

    def __repr__(self):
        return (
            f"SelectionResult(status={self.status!r}, "
            f"selected={list(self.selected)!r}, total_fee={self.total_fee}, "
            f"total_weight={self.total_weight}, nodes_used={self.nodes_used})"
        )


def _check_int(value, name, loc):
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValidationError(
            f"{name} must be an int (bool rejected); got "
            f"{type(value).__name__}",
            loc,
            kind="type_error",
        )


def select_package(pool, block_weight, *, node_budget=DEFAULT_BUDGET):
    """Compute the best dependency-closed package fitting ``block_weight``.

    Args:
        pool: an :class:`AncestorPool`.
        block_weight: positive ``int`` weight capacity.
        node_budget: positive ``int``; maximum number of DFS nodes
            expanded (each costs exactly one unit). Never implicit:
            pass a large int for effectively unbounded search on the
            <=30-tx interface.

    Returns:
        SelectionResult.
    """
    if not isinstance(pool, AncestorPool):
        raise ValidationError(
            f"pool must be an AncestorPool; got {type(pool).__name__}",
            "pool",
            kind="type_error",
        )
    _check_int(block_weight, "block_weight", "block_weight")
    if block_weight <= 0:
        raise ValidationError(
            f"block_weight must be positive; got {block_weight}",
            "block_weight",
        )
    _check_int(node_budget, "node_budget", "node_budget")
    if node_budget <= 0:
        raise ValidationError(
            f"node_budget must be positive; got {node_budget}",
            "node_budget",
        )

    txs, parents = pool.edge_maps()
    n = len(txs)
    if n == 0:
        return SelectionResult(
            (), (), 0, 0, STATUS_PROVEN_OPTIMAL, 0, node_budget, block_weight
        )

    ancestors, _desc = closure_maps(txs, parents)
    order = topo_order(txs, parents)  # DFS decision order
    weight = [txs[tid].weight for tid in order]
    fee = [txs[tid].fee for tid in order]
    par = [
        frozenset(
            k for k, anc_id in enumerate(order) if anc_id in ancestors[tid]
        )
        for tid in order
    ]
    ids_sorted = sorted(txs)

    # Descending fee/weight order, exact cross multiplication; ties by id.
    def ratio_cmp(a, b):
        lhs = fee[a] * weight[b]
        rhs = fee[b] * weight[a]
        if lhs != rhs:
            return -1 if lhs > rhs else 1
        return -1 if order[a] < order[b] else (
            1 if order[a] > order[b] else 0
        )

    ratio_order = sorted(range(n), key=cmp_to_key(ratio_cmp))

    budget = node_budget
    counters = {"nodes": 0, "exhausted": False}
    state = [-1] * n  # -1 undecided, 0 excluded, 1 selected

    best_fee = 0
    best_tuple = ()  # ascending-id tuple of the incumbent

    def consider(sel, total_fee):
        nonlocal best_fee, best_tuple
        tup = tuple(tid for tid in ids_sorted if tid in sel)
        if total_fee > best_fee or (
            total_fee == best_fee and tup < best_tuple
        ):
            best_fee = total_fee
            best_tuple = tup

    def relaxation_bound(i, cap_left, cur_fee):
        """Integer floor of the fractional-knapsack bound on remaining fee.

        Closure between undecided nodes is ignored, so the relaxed value
        is always >= every feasible completion's fee; flooring keeps the
        inequality admissible for integer optima.
        """
        bound = cur_fee
        cap = cap_left
        for j in ratio_order:
            if j < i or state[j] != -1:
                continue
            w = weight[j]
            if w <= cap:
                cap -= w
                bound += fee[j]
                if cap == 0:
                    break
            else:
                bound += (fee[j] * cap) // w
                break
        return bound

    def dfs(i, sel, cur_weight, cur_fee):
        if counters["exhausted"]:
            return
        if counters["nodes"] >= budget:
            # The next expansion would exceed the explicit budget; the
            # optimum is therefore not proved by this call.
            counters["exhausted"] = True
            return
        counters["nodes"] += 1

        if i == n:
            consider(sel, cur_fee)
            return

        # Strict prune only: an equal bound may still hide a
        # lexicographically better tie.
        if relaxation_bound(i, block_weight - cur_weight, cur_fee) < best_fee:
            return

        node = i
        if any(state[p] == 0 for p in par[node]):
            state[node] = 0
            dfs(i + 1, sel, cur_weight, cur_fee)
            state[node] = -1
            return

        fits = cur_weight + weight[node] <= block_weight

        # Select branch first to build a strong incumbent quickly.
        if fits:
            state[node] = 1
            sel.add(order[node])
            dfs(
                i + 1,
                sel,
                cur_weight + weight[node],
                cur_fee + fee[node],
            )
            sel.discard(order[node])
            state[node] = -1
            if counters["exhausted"]:
                return

        state[node] = 0
        dfs(i + 1, sel, cur_weight, cur_fee)
        state[node] = -1

    # Initial feasible incumbent: repeatedly take the highest-ratio
    # node whose parents are already chosen and that still fits.
    cap = block_weight
    g_fee = 0
    chosen = [False] * n
    remaining = set(range(n))
    greedy_sel = set()
    while remaining:
        pick = None
        for j in ratio_order:
            if (
                j in remaining
                and all(chosen[p] for p in par[j])
                and weight[j] <= cap
            ):
                pick = j
                break
        if pick is None:
            break
        chosen[pick] = True
        remaining.discard(pick)
        greedy_sel.add(order[pick])
        cap -= weight[pick]
        g_fee += fee[pick]
    consider(greedy_sel, g_fee)

    dfs(0, set(), 0, 0)

    chosen_ids = set(best_tuple)
    total_fee = sum(txs[t].fee for t in chosen_ids)
    total_weight = sum(txs[t].weight for t in chosen_ids)
    # Lexicographically smallest topological order *of the chosen subset*
    # (filtering the full-graph order would not give that guarantee).
    selected_topo = topo_order(txs, parents, subset=chosen_ids)
    chosen_idx = {j for j in range(n) if order[j] in chosen_ids}
    excluded = _build_exclusions(
        order, par, chosen_idx, block_weight, weight,
        counters["exhausted"],
    )
    status = (
        STATUS_BUDGET_EXHAUSTED
        if counters["exhausted"]
        else STATUS_PROVEN_OPTIMAL
    )
    return SelectionResult(
        selected_topo,
        excluded,
        total_fee,
        total_weight,
        status,
        counters["nodes"],
        node_budget,
        block_weight,
    )


def _build_exclusions(order, par, chosen_idx, block_weight, weight, exhausted):
    exclusions = []
    for j, tid in enumerate(order):
        if j in chosen_idx:
            continue
        missing_anc = [order[p] for p in sorted(par[j]) if p not in chosen_idx]
        if weight[j] > block_weight:
            exclusions.append(
                Exclusion(
                    tid,
                    CODE_OVERWEIGHT,
                    f"weight {weight[j]} exceeds block_weight "
                    f"{block_weight} even without ancestors",
                )
            )
        elif missing_anc:
            exclusions.append(
                Exclusion(
                    tid,
                    CODE_ANCESTOR_EXCLUDED,
                    f"ancestor(s) not in package: {missing_anc}",
                )
            )
        elif exhausted:
            exclusions.append(
                Exclusion(
                    tid,
                    CODE_BUDGET_INCONCLUSIVE,
                    "node budget exhausted before its exclusion could be "
                    "proved necessary",
                )
            )
        else:
            exclusions.append(
                Exclusion(
                    tid,
                    CODE_NOT_SELECTED,
                    "fits together with all its ancestors but is not part "
                    "of the fee-optimal, lexicographically tie-broken "
                    "package",
                )
            )
    return exclusions
