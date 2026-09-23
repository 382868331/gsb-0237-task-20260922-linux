"""祖先包约束的交易池与冲突替换（RPF 风格的有限本题子集）。

``TxPool.add(tx)`` 语义（本题契约，非完整 BIP125）：

1. 结构校验失败（含未知依赖、环、id 重复、非有限数等）：池状态不变，抛错。
2. ``tx.txid`` 已在池中：抛 ``ReplacementRejectedError("already_in_pool")``，
   池不变。
3. 冲突集合 C：池中与新交易消耗 *相同 outpoint* 的交易。
4. 待移除集合 R = C 中每笔交易的全体后代（含其自身）。
5. 拒绝条件（任一成立即整体拒绝，池保持原子不变）：
   * ``depends_on_removed``：新交易显式依赖 R 中的交易；
   * ``fee_not_strictly_greater``：``tx.fee <= sum(fee(r) for r in R)``；
   * ``feerate_not_higher``：``tx.fee/tx.weight <= sumR_fee/sumR_weight``
     （整数交叉相乘比较，严格大于才通过；R 为空时不比较费率）；
   * ``pool_full``：池已达 30 笔上限且本次加入后不净减少
     （即 len(pool) == 30 且 R 为空）。
6. 接受：原子移除 R、加入新交易；新交易不依赖被移除集合后，其余池交易的
   显式依赖仍全部在池中（只有 R 的成员依赖 R，随替换一起移除）。

替换后可能出现“孤儿”：池中某交易消耗的 outpoint（``(creator_txid, vout)``）
其创建交易已在替换中被移除，而该交易没有对创建者的显式 ``depends_on``
（真实系统中替换策略通常禁止这种情况；本题契约允许残留并显式观测）。
``select`` 会把孤儿交易及其 DAG 后代强制排除并给出理由。
"""

from __future__ import annotations

from dataclasses import dataclass

from .errors import (
    InvalidArgumentError,
    PoolFullError,
    ReplacementRejectedError,
)
from .model import MAX_TRANSACTIONS, Transaction
from .solver import (
    EXCL_MISSING_INPUT,
    SelectionResult,
    build_dag,
    solve,
)

# 替换拒绝原因代码
REJ_ALREADY_IN_POOL = "already_in_pool"
REJ_DEPENDS_ON_REMOVED = "depends_on_removed"
REJ_FEE_NOT_GREATER = "fee_not_strictly_greater"
REJ_FEERATE_NOT_HIGHER = "feerate_not_higher"


@dataclass(frozen=True, slots=True)
class AddResult:
    """``add`` 成功的结果。

    Attributes:
        txid: 新加入交易 id。
        removed: 因替换被移除的交易 id（拓扑序：被移除冲突交易及其全部后代）。
        pool_size: 操作后的池大小。
    """

    txid: str
    removed: tuple[str, ...]
    pool_size: int


class TxPool:
    """内存中的合成交易池（有状态、操作原子）。"""

    def __init__(self) -> None:
        self._txs: dict[str, Transaction] = {}
        # 曾在池中、后被替换移除的交易 id；用于精确识别“替换后孤儿”
        # （外部已存在 outpoint 的创建者从不在池中，不算孤儿）。
        self._removed_txids: set[str] = set()

    # ---- 基本查询 -----------------------------------------------------

    def __len__(self) -> int:
        return len(self._txs)

    def __contains__(self, txid: object) -> bool:
        return isinstance(txid, str) and txid in self._txs

    def get(self, txid: str) -> Transaction:
        if not isinstance(txid, str):
            raise InvalidArgumentError(
                f"txid 需要 str，得到 {type(txid).__name__}", field="txid"
            )
        try:
            return self._txs[txid]
        except KeyError:
            raise InvalidArgumentError(f"池中不存在交易 {txid!r}", field="txid")

    def transactions(self) -> tuple[Transaction, ...]:
        """池中全部交易（按 txid 升序的确定性快照）。"""
        return tuple(sorted(self._txs.values(), key=lambda t: t.txid))

    def size(self) -> int:
        return len(self._txs)

    # ---- 核心：add / 替换 ---------------------------------------------

    def add(self, tx: Transaction) -> AddResult:
        """加入一笔交易；冲突时按模块文档字符串的规则原子替换。"""
        if not isinstance(tx, Transaction):
            raise InvalidArgumentError(
                f"需要 Transaction，得到 {type(tx).__name__}", field="tx"
            )
        # Transaction 构造时已做数值/结构校验；这里做池级语义校验。
        if tx.txid in self._txs:
            raise ReplacementRejectedError(
                REJ_ALREADY_IN_POOL,
                f"交易 {tx.txid!r} 已在池中，同 id 替换不被允许",
                txid=tx.txid,
            )

        # 未知依赖（含“依赖一个未在池中的 id”）直接拒绝，池不变。
        missing = sorted(d for d in tx.depends_on if d not in self._txs)
        if missing:
            from .errors import UnknownDependencyError

            raise UnknownDependencyError(
                f"新交易 {tx.txid!r} 依赖池中不存在的 id: {missing}",
                tx_id=tx.txid,
                field="depends_on",
            )

        # 冲突集合 C：相同 outpoint（无需 DAG 即可判定）。
        conflicts: set[str] = set()
        for op in tx.inputs:
            for other in self._txs.values():
                if op in other.inputs:
                    conflicts.add(other.txid)

        # 池上限：无冲突的纯新增在满员时拒绝；有替换时移除>=1、加入1，
        # 不会超限，故允许在满员时继续。
        if not conflicts and len(self._txs) >= MAX_TRANSACTIONS:
            raise PoolFullError(
                f"交易池已达上限 {MAX_TRANSACTIONS}，且本次加入不替换任何交易"
            )

        # 用“现有池 + 新交易”的投影 DAG（至多 31 个临时节点）计算后代集合；
        # 校验失败则池不变（尚未修改任何状态）。
        dag = self._projected_dag(tx)
        new_idx = dag.index[tx.txid]

        if new_idx in dag.ancestors[new_idx]:
            raise InvalidArgumentError(
                f"加入 {tx.txid!r} 后依赖图出现环", field="depends_on"
            )

        # R = C ∪ C 的全部后代（后代只可能是池内交易）。
        removed_idx: set[int] = set()
        for cid in conflicts:
            ci = dag.index[cid]
            removed_idx.add(ci)
            removed_idx.update(dag.descendants[ci])
        removed_idx.discard(new_idx)

        # 拒绝条件 1：新交易依赖 R（显式 depends_on，或消耗 R 中交易创建的
        # outpoint —— 后者是隐式依赖，同样禁止）。
        removed_id_set = {dag.txs[j].txid for j in removed_idx}
        deps_in_removed = sorted(removed_id_set & set(tx.depends_on))
        implicit = sorted(
            {op.txid for op in tx.inputs if op.txid in removed_id_set}
        )
        if deps_in_removed or implicit:
            raise ReplacementRejectedError(
                REJ_DEPENDS_ON_REMOVED,
                "新交易不得依赖将被移除的冲突交易或其后代（显式依赖或消耗其输出）",
                txid=tx.txid,
                removed=sorted(set(deps_in_removed) | set(implicit)),
            )

        # 拒绝条件 2/3：绝对费用严格更大；费率严格更高（整数交叉相乘）。
        if removed_idx:
            removed_fee = sum(dag.txs[j].fee for j in removed_idx)
            removed_weight = sum(dag.txs[j].weight for j in removed_idx)
            if tx.fee <= removed_fee:
                raise ReplacementRejectedError(
                    REJ_FEE_NOT_GREATER,
                    f"新交易 fee={tx.fee} 必须严格大于被移除集合总 fee={removed_fee}",
                    txid=tx.txid,
                    new_fee=tx.fee,
                    removed_fee=removed_fee,
                )
            if tx.fee * removed_weight <= removed_fee * tx.weight:
                raise ReplacementRejectedError(
                    REJ_FEERATE_NOT_HIGHER,
                    (
                        f"新交易费率 {tx.fee}/{tx.weight} 必须严格高于被移除集合费率 "
                        f"{removed_fee}/{removed_weight}"
                    ),
                    txid=tx.txid,
                    new_feerate=(tx.fee, tx.weight),
                    removed_feerate=(removed_fee, removed_weight),
                )
        # 无冲突纯新增的满员情况已在前面拦截。

        # ---- 接受：原子提交（前面的所有计算均未修改 self._txs）。----
        removed_ids = tuple(
            dag.txs[j].txid
            for j in dag.order
            if j in removed_idx and dag.txs[j].txid != tx.txid
        )
        for rid in removed_ids:
            del self._txs[rid]
            self._removed_txids.add(rid)
        self._txs[tx.txid] = tx
        return AddResult(txid=tx.txid, removed=removed_ids, pool_size=len(self._txs))

    # ---- select -------------------------------------------------------

    def select(
        self,
        block_weight: int,
        node_budget: int = 1_000_000,
    ) -> SelectionResult:
        """在当前池快照上求块重内费用最优的依赖闭合包。

        替换后产生的 *孤儿*（消耗的 outpoint 创建者已被移除、且无显式依赖
        声明的池内交易）在搜索前被强制排除，其 DAG 后代随祖先规则一并排除，
        理由代码为 ``missing_input_outpoint`` / ``ancestor_excluded``。
        """
        txs_list = list(self.transactions())
        dag = build_dag(txs_list)
        orphan_map = self._orphan_inputs(dag)
        forced = {
            dag.index[txid]: (
                EXCL_MISSING_INPUT,
                f"消耗的 outpoint 创建交易已被替换移除且无显式依赖: {missing}",
            )
            for txid, missing in orphan_map.items()
        }
        return solve(
            txs_list,
            block_weight,
            node_budget=node_budget,
            forced_excluded=forced,
        )

    # ---- 替换后观测 ---------------------------------------------------

    def orphans(self) -> dict[str, tuple[str, ...]]:
        """替换后孤儿：txid -> 该交易缺失创建者的 outpoint 字符串元组。

        仅统计 *曾在池中后被替换移除* 的创建者（外部 outpoint 不算）；
        且该交易未通过 ``depends_on`` 显式声明（显式声明未知依赖在 add 时即被拒）。
        """
        return dict(self._orphan_inputs(build_dag(list(self.transactions()))))

    def _orphan_inputs(self, dag) -> dict[str, tuple[str, ...]]:
        """内部：基于给定 DAG 计算孤儿根 -> 缺失 outpoint 描述。"""
        txs = dag.txs
        present = {t.txid for t in txs}
        out: dict[str, tuple[str, ...]] = {}
        for t in txs:
            missing = sorted(
                str(op)
                for op in t.inputs
                if op.txid in self._removed_txids
                and op.txid not in present
                and op.txid not in t.depends_on
            )
            if missing:
                out[t.txid] = tuple(missing)
        return out

    # ---- 内部 ---------------------------------------------------------

    def _projected_dag(self, extra: Transaction | None):
        """在“池快照（+可选新交易）”上构造 solver DAG（临时投影豁免 30 上限）。"""
        txs = self.transactions()
        if extra is not None:
            txs = (*txs, extra)
        return build_dag(list(txs), enforce_limit=False)
