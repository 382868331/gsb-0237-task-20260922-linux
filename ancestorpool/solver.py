"""依赖闭合子集的最大手续费求解（祖先包选择）。

问题定义（本题契约）
--------------------
给定交易 DAG（``depends_on`` 边 父 -> 子）、每笔交易的整数 weight/fee 和块重
上限 ``block_weight``，在所有 *依赖闭合* 的子集中选总 fee 最大者：

* 依赖闭合：子集中任一交易的全部 depends_on 祖先也在子集中；
* 可行性：闭合子集总 weight <= block_weight；
* 同分时按“入选 txid 升序排列的元组”字典序最小破同分（确定性、与边顺序无关）。

算法
----
带祖先强制约束的深度优先分支定界（branch & bound），三值状态
（入选 1 / 排除 0 / 未定 -1），按拓扑序处理：

* 入选 i：其全部祖先闭包必须入选（拓扑序保证祖先都已决策，遇已排除祖先则剪枝）；
* 排除 i：其全部后代也必须排除（祖先缺失则后代无法闭合）；
* 闭包超重剪枝：i 连同祖先闭包总重超过块重时，i 不可入选；
* 费用上界剪枝：当前费用 + 全部未定交易费用之和 不优于已知最优时回溯
  （费用非负，上界宽松但严格安全）；
* 先探“入选”分支，尽早得到高费 incumbent，增强剪枝。

显式预算以“搜索节点数”计（``node_budget``，默认 ``DEFAULT_NODE_BUDGET``）。
预算耗尽抛 :class:`SearchBudgetExhaustedError` —— 只代表未能证明最优，
不代表无解。空集永远是可行闭合子集（总重 0），因此本契约下不存在
“已证明无可行解”的情形；正常返回即覆盖全部可行空间后的证明。
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cmp_to_key

from .errors import (
    DuplicateTransactionIdError,
    InvalidArgumentError,
    SearchBudgetExhaustedError,
)
from .model import MAX_TRANSACTIONS, Transaction, checked_positive_int

#: 默认搜索节点预算（每次分支决策计一个节点）。
DEFAULT_NODE_BUDGET = 1_000_000

# 排除原因代码
EXCL_CLOSURE_WEIGHT = "ancestor_closure_overweight"
EXCL_ANCESTOR_EXCLUDED = "ancestor_excluded"
EXCL_MISSING_INPUT = "missing_input_outpoint"
EXCL_NOT_SELECTED = "not_in_optimal_package"


@dataclass(frozen=True, slots=True)
class Exclusion:
    """未入选交易的排除说明。"""

    txid: str
    reason: str
    detail: str


@dataclass(frozen=True, slots=True)
class SelectionResult:
    """``select`` 的结果。

    Attributes:
        selected: 入选交易，按拓扑序输出（父在子前；同层按 txid 升序，确定唯一）。
        total_fee: 入选总手续费（整数）。
        total_weight: 入选总权重（整数）。
        block_weight: 本次块重上限。
        exclusions: 每个未入选交易一条排除说明（按 txid 升序）。
        nodes_visited: 实际访问的搜索节点数（用于观察预算消耗）。
        proven_optimal: 搜索完整覆盖（未耗尽预算）时为 True。
    """

    selected: tuple[str, ...]
    total_fee: int
    total_weight: int
    block_weight: int
    exclusions: tuple[Exclusion, ...]
    nodes_visited: int
    proven_optimal: bool


@dataclass(slots=True)
class _DAG:
    txs: list[Transaction]
    index: dict[str, int]
    parents: list[frozenset[int]]
    children: list[frozenset[int]]
    ancestors: list[frozenset[int]]
    descendants: list[frozenset[int]]
    order: list[int]  # 拓扑序（同层按 txid 升序）


def _reachable(start: int, adj: list[frozenset[int]]) -> frozenset[int]:
    """从 start 沿 adj 求可达集合（不含 start）。"""
    seen: set[int] = set()
    stack = [start]
    while stack:
        u = stack.pop()
        for v in adj[u]:
            if v not in seen:
                seen.add(v)
                stack.append(v)
    return frozenset(seen)


def build_dag(txs: list[Transaction], enforce_limit: bool = True) -> _DAG:
    """从交易列表构造内部 DAG；校验 id 唯一、依赖存在、无环。

    ``enforce_limit=False`` 仅供池内部构造“现有池 + 1 笔候选”（最多 31 节点）
    的临时投影使用；对外的 :func:`solve` 始终强制 30 笔上限。
    """
    n = len(txs)
    if enforce_limit and n > MAX_TRANSACTIONS:
        raise InvalidArgumentError(
            f"交易数 {n} 超过上限 {MAX_TRANSACTIONS}", field="transactions"
        )
    index: dict[str, int] = {}
    for i, tx in enumerate(txs):
        if tx.txid in index:
            raise DuplicateTransactionIdError(
                "交易 id 重复",
                tx_id=tx.txid,
                field=f"txs[{i}].txid",
            )
        index[tx.txid] = i

    parents: list[frozenset[int]] = []
    children_sets: list[set[int]] = [set() for _ in range(n)]
    for i, tx in enumerate(txs):
        ps: set[int] = set()
        for dep in tx.depends_on:
            if dep not in index:
                raise InvalidArgumentError(
                    f"交易 {tx.txid!r} 依赖未知 id {dep!r}",
                    field=f"txs[{i}].depends_on",
                )
            ps.add(index[dep])
        if i in ps:
            raise InvalidArgumentError(
                f"交易 {tx.txid!r} 不能依赖自身", field=f"txs[{i}].depends_on"
            )
        parents.append(frozenset(ps))
        for p in ps:
            children_sets[p].add(i)

    children = [frozenset(c) for c in children_sets]
    ancestors = [_reachable(i, parents) for i in range(n)]
    descendants = [_reachable(i, children) for i in range(n)]

    # 环检测 + 拓扑序（Kahn，同层 txid 升序）。
    indeg = [len(parents[i]) for i in range(n)]
    ready = sorted((i for i in range(n) if indeg[i] == 0), key=lambda i: txs[i].txid)
    order: list[int] = []
    while ready:
        u = ready.pop(0)
        order.append(u)
        for v in sorted(children[u], key=lambda i: txs[i].txid):
            indeg[v] -= 1
            if indeg[v] == 0:
                lo, hi = 0, len(ready)
                while lo < hi:
                    mid = (lo + hi) // 2
                    if txs[ready[mid]].txid < txs[v].txid:
                        lo = mid + 1
                    else:
                        hi = mid
                ready.insert(lo, v)
    if len(order) != n:
        cyc = sorted(txs[i].txid for i in range(n) if i not in set(order))
        raise InvalidArgumentError(
            f"依赖图存在环，涉及交易: {cyc}", field="depends_on"
        )

    return _DAG(txs, index, parents, children, ancestors, descendants, order)


def solve(
    txs: list[Transaction],
    block_weight: int,
    node_budget: int = DEFAULT_NODE_BUDGET,
    forced_excluded: dict[int, tuple[str, str]] | None = None,
) -> SelectionResult:
    """求依赖闭合、重量受限的最大 fee 子集。详见模块文档字符串。

    ``forced_excluded``: ``{节点下标: (reason_code, detail)}``，这些交易在搜索
    开始前即被强制排除（其后代随否决传播一并排除），排除理由原样输出。
    """
    bw = checked_positive_int(block_weight, "block_weight")
    budget = checked_positive_int(node_budget, "node_budget")
    dag = build_dag(txs)
    n = len(dag.txs)
    ord_nodes = dag.order
    weights = [tx.weight for tx in txs]
    fees = [tx.fee for tx in txs]

    # 每个交易连同祖先闭包的总重；超重者永不可入选。
    closure_w = [
        weights[i] + sum(weights[a] for a in dag.ancestors[i]) for i in range(n)
    ]

    # 状态位：1=入选，0=排除，-1=未定。先落强制排除（含其后代传播）。
    state = [-1] * n
    forced_reasons: dict[int, tuple[str, str]] = dict(forced_excluded or {})
    root_forced: set[int] = set()
    for idx in forced_reasons:
        if not 0 <= idx < n:
            raise InvalidArgumentError(
                f"forced_excluded 下标 {idx} 越界", field="forced_excluded"
            )
        root_forced.add(idx)
    propagated: set[int] = set()
    for r in root_forced:
        for d in dag.descendants[r]:
            propagated.add(d)
    pre_excluded = root_forced | propagated

    for j in pre_excluded:
        state[j] = 0

    best_mask = 0
    best_fee = 0
    best_weight = 0
    best_ids: tuple[str, ...] = ()
    nodes = 0

    # 按 txid 升序排列的节点（字典序同分判定与分支顺序）。
    by_txid = sorted(range(n), key=lambda j: txs[j].txid)

    # 全局费率降序（费率相等按 txid 升序）；交叉相乘排序，只算一次。
    def _rate_cmp(a: int, b: int) -> int:
        lhs = fees[a] * weights[b]
        rhs = fees[b] * weights[a]
        if lhs != rhs:
            return -1 if lhs > rhs else 1
        return -1 if txs[a].txid < txs[b].txid else (
            1 if txs[a].txid > txs[b].txid else 0
        )

    rate_order = sorted(range(n), key=cmp_to_key(_rate_cmp))

    def sorted_ids(mask: int) -> tuple[str, ...]:
        return tuple(txs[j].txid for j in by_txid if (mask >> j) & 1)

    def fractional_ub(capacity: int, gap: int) -> int:
        """未定交易（state==-1）上的分数背包松弛上界，与 gap 精确比较。

        忽略闭合约束，故为合法松弛。返回 -1/0/1 分别表示上界
        严格小于 / 等于 / 严格大于 gap。全整数交叉相乘，无浮点。
        """
        cap = capacity
        int_fee = 0
        total_undecided_w = 0
        for j in rate_order:
            if state[j] != -1:
                continue
            w, f = weights[j], fees[j]
            total_undecided_w += w
            if cap <= 0:
                continue
            if w <= cap:
                cap -= w
                int_fee += f
            else:
                # 最后一个分数项：有理上界 int_fee + cap*f/w。
                lhs = (int_fee - gap) * w + cap * f
                return -1 if lhs < 0 else (0 if lhs == 0 else 1)
        if total_undecided_w <= capacity:
            # 全部整数装下，上界精确。
            if int_fee < gap:
                return -1
            return 0 if int_fee == gap else 1
        # cap 提前耗尽且无分数项：上界即 int_fee。
        return -1 if int_fee < gap else (0 if int_fee == gap else 1)

    def lex_tied_subtree_prunable(pos: int) -> bool:
        """费用上界只能追平最优时，判断本子树是否不可能字典序更优。

        C = 当前已入选（state==1，含更深层强制加入的祖先）的 txid 升序元组；
        子树任意解为 C ∪ S，S 为未定节点子集；pos 指向最小 txid 的未定节点。
        """
        c_ids = [txs[j].txid for j in by_txid if state[j] == 1]
        m = min(len(c_ids), len(best_ids))
        k = 0
        while k < m and c_ids[k] == best_ids[k]:
            k += 1
        if k < m:
            # 首个分歧元素决定整棵子树胜负。
            return c_ids[k] > best_ids[k]
        if len(c_ids) >= len(best_ids):
            # C 更长（更差）或完全相同（再加入任何 id 都更长）。
            return True
        # B 还有后续元素；本侧首个新增 id 必须严格小于该阈值。
        # 等于阈值时该 id 自身可能入选并在更后方再分胜负，故不能剪。
        threshold = best_ids[len(c_ids)]
        return txs[by_txid[pos]].txid > threshold

    def search(pos: int, cur_mask: int, cur_w: int, cur_f: int) -> None:
        """pos 为 by_txid 中最小的未定扫描位置（不变量：其前均已决策）。"""
        nonlocal best_mask, best_fee, best_weight, best_ids, nodes
        nodes += 1
        if nodes > budget:
            raise SearchBudgetExhaustedError(
                "select 搜索节点预算耗尽，无法证明最优",
                nodes_visited=nodes,
                node_budget=budget,
            )

        # 当前状态（未定者视为不选）是合法闭合包（祖先始终随入选强制加入）。
        if cur_f > best_fee or (
            cur_f == best_fee and sorted_ids(cur_mask) < best_ids
        ):
            best_mask, best_fee, best_weight = cur_mask, cur_f, cur_w
            best_ids = sorted_ids(cur_mask)

        # 找最小 txid 的未定节点。
        while pos < n and state[by_txid[pos]] != -1:
            pos += 1
        if pos >= n:
            return

        cmp_ub = fractional_ub(bw - cur_w, best_fee - cur_f)
        if cmp_ub < 0:
            return
        if cmp_ub == 0 and lex_tied_subtree_prunable(pos):
            return

        i = by_txid[pos]
        bit = 1 << i

        # 分支 1（优先）：入选 i，祖先闭包中未定者一并强制入选。
        if (
            closure_w[i] <= bw
            and cur_w + weights[i] <= bw
            and all(state[a] != 0 for a in dag.ancestors[i])
        ):
            forced = [a for a in dag.ancestors[i] if state[a] == -1]
            fw = sum(weights[a] for a in forced)
            if cur_w + weights[i] + fw <= bw:
                ff = sum(fees[a] for a in forced)
                mb = cur_mask | bit
                state[i] = 1
                for a in forced:
                    state[a] = 1
                    mb |= 1 << a
                search(pos + 1, mb, cur_w + weights[i] + fw, cur_f + fees[i] + ff)
                state[i] = -1
                for a in forced:
                    state[a] = -1

        # 分支 2：排除 i，其全部未定后代随祖先缺失一并排除。
        killed = [i]
        state[i] = 0
        for d in dag.descendants[i]:
            if state[d] == -1:
                state[d] = 0
                killed.append(d)
        search(pos + 1, cur_mask, cur_w, cur_f)
        for d in killed:
            state[d] = -1

    search(0, 0, 0, 0)

    selected = tuple(txs[i].txid for i in ord_nodes if (best_mask >> i) & 1)
    sel_set = {dag.index[t] for t in selected}

    exclusions: list[Exclusion] = []
    for i in sorted(range(n), key=lambda j: txs[j].txid):
        if i in sel_set:
            continue
        if i in root_forced:
            code, detail = forced_reasons[i]
            exclusions.append(Exclusion(txs[i].txid, code, detail))
            continue
        if i in propagated:
            miss = sorted(
                txs[a].txid
                for a in dag.ancestors[i]
                if a in pre_excluded and a not in sel_set
            )
            exclusions.append(
                Exclusion(
                    txs[i].txid,
                    EXCL_ANCESTOR_EXCLUDED,
                    f"祖先被强制排除，依赖无法闭合: {miss}",
                )
            )
            continue
        if closure_w[i] > bw:
            exclusions.append(
                Exclusion(
                    txs[i].txid,
                    EXCL_CLOSURE_WEIGHT,
                    f"该交易连同祖先闭包总重 {closure_w[i]} 超过块重 {bw}",
                )
            )
            continue
        miss = sorted(txs[a].txid for a in dag.ancestors[i] if a not in sel_set)
        if miss:
            exclusions.append(
                Exclusion(
                    txs[i].txid,
                    EXCL_ANCESTOR_EXCLUDED,
                    f"祖先未入选，依赖无法闭合: {miss}",
                )
            )
            continue
        exclusions.append(
            Exclusion(
                txs[i].txid,
                EXCL_NOT_SELECTED,
                "块重约束下存在手续费不更低且字典序不更差的其他闭合包",
            )
        )

    return SelectionResult(
        selected=selected,
        total_fee=best_fee,
        total_weight=best_weight,
        block_weight=bw,
        exclusions=tuple(exclusions),
        nodes_visited=nodes,
        proven_optimal=True,
    )
