# ============================================================================
# 文件: factors/__init__.py
# ============================================================================
"""
因子模块

重要: 必须导入所有因子子模块，触发 @FactorRegistry.register 装饰器
"""

# 基类和注册表
from .base import BaseFactor, FactorMeta, FactorRegistry, FactorPipeline

# ============================================================
# 关键: 导入所有因子模块，触发装饰器注册
# ============================================================

# 技术因子
from .technical import rsrs
from .technical import momentum
from .technical import volatility

# 高级 RSRS 因子
try:
    from .technical import rsrs_advanced
except ImportError:
    pass

# 复合因子
try:
    from .composite import alpha_score
except ImportError:
    pass

# Alpha Hunter 因子
try:
    from . import alpha_hunter_factors
except ImportError:
    pass

# Alpha Hunter V2 (新增)
try:
    from . import alpha_hunter_v2_factors
except ImportError:
    pass

# ============================================================
# 导出
# ============================================================
__all__ = [
    'BaseFactor',
    'FactorMeta',
    'FactorRegistry',
    'FactorPipeline',
]