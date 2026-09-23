"""公开数据结构：``Outpoint``、``Transaction``。

数值契约
--------
* ``weight`` 为正整数；``fee`` 为非负整数（以 sat 为单位的合成数据）。
  整数参数拒绝 ``bool``（``True``/``False`` 不被当作 1/0）。
* 任何 float 若进入数值字段，必须有限（拒绝 NaN/±Infinity）；本库数值字段
  只接受 int，float 一律报类型错误——避免浮点费率比较引入边界误差。
* ``block_weight`` 同样为正整数（非 bool）。费率比较全部使用整数交叉相乘，
  不存在浮点误差（误差口径为 0）。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from .errors import InvalidArgumentError, InvalidTransactionError, NonFiniteNumberError

#: 接口边界：单次 select 可见的池内交易最多 30 笔。
MAX_TRANSACTIONS = 30


def reject_nonfinite(value: Any, field_name: str) -> None:
    """若 value 是 NaN/±Infinity 的 float，抛 :class:`NonFiniteNumberError`。"""
    if isinstance(value, float) and not math.isfinite(value):
        raise NonFiniteNumberError(
            "数值必须有限，禁止 NaN/Infinity", field=field_name
        )


def checked_positive_int(value: Any, field_name: str) -> int:
    """校验为“正整数且非 bool”。"""
    reject_nonfinite(value, field_name)
    if isinstance(value, bool) or not isinstance(value, int):
        raise InvalidArgumentError(
            f"需要正整数 int，得到 {type(value).__name__}", field=field_name
        )
    if value <= 0:
        raise InvalidArgumentError(
            f"需要正整数，得到 {value}", field=field_name
        )
    return value


def checked_nonneg_int(value: Any, field_name: str) -> int:
    """校验为“非负整数且非 bool”。"""
    reject_nonfinite(value, field_name)
    if isinstance(value, bool) or not isinstance(value, int):
        raise InvalidArgumentError(
            f"需要非负整数 int，得到 {type(value).__name__}", field=field_name
        )
    if value < 0:
        raise InvalidArgumentError(
            f"需要非负整数，得到 {value}", field=field_name
        )
    return value


def checked_str(value: Any, field_name: str) -> str:
    if not isinstance(value, str):
        raise InvalidArgumentError(
            f"需要非空 str，得到 {type(value).__name__}", field=field_name
        )
    if not value:
        raise InvalidArgumentError("字符串不能为空", field=field_name)
    return value


@dataclass(frozen=True, slots=True)
class Outpoint:
    """被消耗的交易输出点：``(txid, vout)``。

    合成数据中 txid 只要求是非空 str；vout 为非负整数（非 bool）。
    """

    txid: str
    vout: int

    def __post_init__(self) -> None:
        # frozen dataclass 下用 object.__setattr__ 无必要，只做校验。
        checked_str(self.txid, "outpoint.txid")
        checked_nonneg_int(self.vout, "outpoint.vout")

    def __str__(self) -> str:
        return f"{self.txid}:{self.vout}"


@dataclass(frozen=True, slots=True)
class Transaction:
    """一笔合成交易。

    Attributes:
        txid: 唯一交易 id（非空 str）。
        weight: 权重，正整数。
        fee: 手续费，非负整数。
        inputs: 消耗的 outpoint 集合（去重，空集合表示无人可与之双花的“铸造型”
            交易，仍可通过 depends_on 建立依赖）。
        depends_on: 显式依赖的父交易 txid；选中某交易时其全部依赖必须同时入选
            （祖先包 / 依赖闭合）。
    """

    txid: str
    weight: int
    fee: int
    inputs: frozenset[Outpoint] = frozenset()
    depends_on: frozenset[str] = frozenset()

    def __init__(
        self,
        txid: str,
        weight: int,
        fee: int,
        inputs: Any = (),
        depends_on: Any = (),
    ) -> None:
        # frozen + slots 时需要经 object.__setattr__；逐字段校验后赋值。
        txid_v = checked_str(txid, "txid")
        weight_v = checked_positive_int(weight, "weight")
        fee_v = checked_nonneg_int(fee, "fee")

        if not isinstance(inputs, (frozenset, set, tuple, list)):
            raise InvalidTransactionError(
                f"inputs 需要可迭代集合，得到 {type(inputs).__name__}",
                tx_id=txid_v if isinstance(txid_v, str) else None,
                field="inputs",
            )
        norm_inputs: set[Outpoint] = set()
        for i, op in enumerate(inputs):
            if not isinstance(op, Outpoint):
                raise InvalidTransactionError(
                    f"inputs[{i}] 需要 Outpoint，得到 {type(op).__name__}",
                    tx_id=txid_v,
                    field=f"inputs[{i}]",
                )
            norm_inputs.add(op)

        if not isinstance(depends_on, (frozenset, set, tuple, list)):
            raise InvalidTransactionError(
                f"depends_on 需要可迭代的 str 集合，得到 {type(depends_on).__name__}",
                tx_id=txid_v,
                field="depends_on",
            )
        norm_deps: set[str] = set()
        for i, dep in enumerate(depends_on):
            norm_deps.add(checked_str(dep, f"depends_on[{i}]"))

        for f_name, value in (
            ("txid", txid_v),
            ("weight", weight_v),
            ("fee", fee_v),
            ("inputs", frozenset(norm_inputs)),
            ("depends_on", frozenset(norm_deps)),
        ):
            object.__setattr__(self, f_name, value)
