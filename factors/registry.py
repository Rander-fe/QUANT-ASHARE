# -*- coding: utf-8 -*-
"""
因子注册表（Factor Registry）

功能：
    - 统一管理所有因子的元数据：名称、分类、计算函数、注释
    - 提供按分类统计数量的接口
    - 预留复合因子（composite）支持

用法：
    from factors.registry import register, get_factor_list, get_factor_count

    # 在因子模块中注册
    register(
        name="REV5",
        category="reversal_momentum",
        func=calc_rev5,
        comment="5日反转，越高代表过去5日跌幅越大"
    )
"""

from typing import Callable, Dict, List, Optional

# 因子注册表结构：
# {
#   "factor_name": {
#       "category": str,
#       "func": Callable,
#       "comment": str,
#   }
# }
_FACTOR_REGISTRY: Dict[str, Dict] = {}

# 允许的分类列表（与计划一致）
ALLOWED_CATEGORIES = [
    "reversal_momentum",   # 反转/动量
    "volatility",          # 波动率
    "liquidity",           # 流动性
    "technical",           # 技术指标
    "value",               # 估值
    "quality",             # 质量
    "growth",              # 成长
    "composite",           # 复合因子（预留）
]


def register(
    name: str,
    category: str,
    func: Callable,
    comment: Optional[str] = None,
) -> None:
    """
    注册一个因子到全局注册表。

    Parameters:
        name: 因子名称（必须唯一，建议格式如 REV5、ROE_TTM）
        category: 因子分类，必须是 ALLOWED_CATEGORIES 之一
        func: 因子计算函数，签名为 func(df: pd.DataFrame) -> pd.Series
        comment: 因子注释（说明计算逻辑或业务含义）
    """
    if category not in ALLOWED_CATEGORIES:
        raise ValueError(
            f"Unknown category '{category}'. Allowed: {ALLOWED_CATEGORIES}"
        )
    if name in _FACTOR_REGISTRY:
        raise ValueError(f"Factor '{name}' already registered!")

    _FACTOR_REGISTRY[name] = {
        "category": category,
        "func": func,
        "comment": comment or "",
    }


def get_factor(name: str) -> Optional[Dict]:
    """获取单个因子的注册信息"""
    return _FACTOR_REGISTRY.get(name)


def get_factor_list(category: Optional[str] = None) -> List[str]:
    """
    获取因子名称列表。

    Parameters:
        category: 若指定，只返回该分类下的因子名称
    """
    if category is None:
        return list(_FACTOR_REGISTRY.keys())
    return [
        name for name, info in _FACTOR_REGISTRY.items()
        if info["category"] == category
    ]


def get_factor_count(category: Optional[str] = None) -> int:
    """获取因子数量"""
    return len(get_factor_list(category))


def get_category_summary() -> Dict[str, int]:
    """返回各分类的因子数量统计"""
    summary = {cat: 0 for cat in ALLOWED_CATEGORIES}
    for info in _FACTOR_REGISTRY.values():
        summary[info["category"]] += 1
    return summary


def print_registry_summary() -> None:
    """打印注册表摘要（用于验证总数 ≥ 100）"""
    print("\n" + "=" * 60)
    print("📋 因子注册表摘要")
    print("=" * 60)
    summary = get_category_summary()
    total = 0
    for cat, count in summary.items():
        if count > 0:
            print(f"   {cat:20s}: {count:3d}")
        total += count
    print("-" * 60)
    print(f"   TOTAL{'':20s}: {total:3d}")
    print("=" * 60)