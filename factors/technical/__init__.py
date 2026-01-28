# ============================================================================
# 文件: factors/technical/__init__.py
# ============================================================================
"""
技术因子模块

导入所有因子类以触发注册
"""

from .rsrs import RSRSFactor, RSRSZScoreFactor, RSRSValidFactor
from .momentum import (
    MomentumFactor,
    ROCFactor,
    OBVTrendFactor,
    VWAPBiasFactor,
    VolumeRankFactor
)
from .volatility import (
    ATRFactor,
    ATRPercentFactor,
    VolatilityFactor,
    ChandelierStopFactor,
    VolatilityRegimeFactor
)

# 高级 RSRS (如果存在)
try:
    from .rsrs_advanced import (
        RSRSAdvancedFactor,
        RSRSMomentumFactor,
        RSRSMultiPeriodFactor
    )
except ImportError:
    pass

__all__ = [
    # RSRS
    'RSRSFactor', 'RSRSZScoreFactor', 'RSRSValidFactor',
    # Momentum
    'MomentumFactor', 'ROCFactor', 'OBVTrendFactor',
    'VWAPBiasFactor', 'VolumeRankFactor',
    # Volatility
    'ATRFactor', 'ATRPercentFactor', 'VolatilityFactor',
    'ChandelierStopFactor', 'VolatilityRegimeFactor',
]