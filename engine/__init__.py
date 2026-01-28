# ============================================================================
# 文件: engine/__init__.py
# ============================================================================
"""引擎模块"""

from .matcher import MatchEngine, Order, OrderStatus
from .portfolio import PortfolioManager, Position
from .backtest import BacktestEngine, BacktestResult

# 高频撮合器
try:
    from .high_freq_matcher import HighFreqMatcher
except ImportError:
    pass

# V2 高频撮合器 (新增)
try:
    from .high_freq_matcher_v2 import HighFreqMatcherV2, LimitAnalyzer
except ImportError:
    pass

__all__ = [
    'MatchEngine', 'Order', 'OrderStatus',
    'PortfolioManager', 'Position',
    'BacktestEngine', 'BacktestResult'
]