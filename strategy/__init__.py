# ============================================================================
# 文件: strategy/__init__.py
# ============================================================================
"""策略模块"""

from .base import BaseStrategy, Signal, OrderSide, StrategyContext
from .registry import StrategyRegistry

# 导入策略以触发注册
from . import rsrs_strategy
from . import momentum_strategy

# 可选策略
try:
    from . import short_term_strategy
except ImportError:
    pass

try:
    from . import alpha_hunter_strategy
except ImportError:
    pass

# Alpha Hunter V2 (新增)
try:
    from . import alpha_hunter_v2_strategy
except ImportError:
    pass

__all__ = [
    'BaseStrategy',
    'Signal',
    'OrderSide',
    'StrategyContext',
    'StrategyRegistry'
]