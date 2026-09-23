"""错误类型与错误语义。

所有库异常都继承 :class:`AncestorPoolError`，调用方可按基类统一捕获。
错误信息中带有可定位字段（字段名、交易 id、outpoint 等），库不做静默纠正。
"""

from __future__ import annotations


class AncestorPoolError(Exception):
    """本库所有异常的基类。"""


class InvalidArgumentError(AncestorPoolError, ValueError):
    """通用参数非法（类型错误、越界、NaN/Infinity 等）。

    ``field`` 给出可定位的字段路径（如 ``"weight"``、``"txs[2].fee"``）。
    """

    def __init__(self, message: str, field: str | None = None) -> None:
        self.field = field
        if field is not None:
            message = f"{message} [field={field}]"
        super().__init__(message)


class NonFiniteNumberError(InvalidArgumentError):
    """数值为 NaN 或 ±Infinity。"""


class InvalidTransactionError(InvalidArgumentError):
    """单笔交易结构/取值非法。"""

    def __init__(self, message: str, tx_id: str | None = None, field: str | None = None) -> None:
        self.tx_id = tx_id
        if tx_id is not None and field is not None:
            message = f"{message} [tx_id={tx_id!r}, field={field}]"
        elif tx_id is not None:
            message = f"{message} [tx_id={tx_id!r}]"
        elif field is not None:
            message = f"{message} [field={field}]"
        # 不调用 super().__init__，避免双重追加 field 后缀。
        ValueError.__init__(self, message)


class DuplicateTransactionIdError(InvalidTransactionError):
    """交易 id 在同一批 add 中重复，或池中已存在该 id。"""


class UnknownDependencyError(InvalidTransactionError):
    """交易依赖（depends_on）指向未知 id。"""


class PoolFullError(AncestorPoolError):
    """池大小已达 ``MAX_TRANSACTIONS``，且本次 add 不净减少池大小。"""


class ReplacementRejectedError(AncestorPoolError):
    """冲突替换的策略条件不满足，或交易会被挂起（缺依赖/双花）。

    ``reason`` 为机器可读的原因代码，``details`` 携带被移除集合等上下文。
    """

    def __init__(self, reason: str, message: str | None = None, **details: object) -> None:
        self.reason = reason
        self.details = details
        text = reason if message is None else f"{reason}: {message}"
        if details:
            kv = ", ".join(f"{k}={v!r}" for k, v in details.items())
            text = f"{text} ({kv})"
        super().__init__(text)


class SearchBudgetExhaustedError(AncestorPoolError):
    """``select`` 搜索预算耗尽，最优性无法证明。

    与“已证明无可行解”严格区分：预算耗尽只代表搜索没做完，不代表无解。
    """

    def __init__(self, message: str, *, nodes_visited: int, node_budget: int) -> None:
        self.nodes_visited = nodes_visited
        self.node_budget = node_budget
        super().__init__(f"{message} (nodes_visited={nodes_visited}, node_budget={node_budget})")
