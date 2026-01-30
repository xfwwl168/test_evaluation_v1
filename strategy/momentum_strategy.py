# ============================================================================
# 文件: strategy/momentum_strategy.py
# ============================================================================
"""
动量策略实现 - 基于价格动量和趋势跟随
"""
import numpy as np
import pandas as pd
from typing import Dict, List

from .base import BaseStrategy, Signal, OrderSide, StrategyContext
from .registry import StrategyRegistry
from factors import FactorRegistry
from config import settings


@StrategyRegistry.register
class MomentumStrategy(BaseStrategy):
    """
    动量策略
    
    核心逻辑:
    - 选择过去 N 日涨幅最大的股票
    - 结合波动率进行仓位调整
    - 使用均值回归作为离场信号
    
    参数:
    - lookback: 动量回溯周期
    - top_n: 选股数量
    - rebalance_threshold: 调仓阈值
    """
    
    name = "momentum"
    version = "1.0.0"
    
    def __init__(self, params: Dict = None):
        default_params = {
            'lookback': settings.factor.MOMENTUM_WINDOW,
            'top_n': 20,
            'min_momentum': 0.05,      # 最低动量 5%
            'max_volatility': 0.5,     # 最大波动率 50%
            'use_volatility_weight': True,
        }
        
        merged_params = {**default_params, **(params or {})}
        super().__init__(merged_params)
    
    def compute_factors(self, history: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """计算因子"""
        lookback = self.get_param('lookback')
        
        momentum_factor = FactorRegistry.get("momentum", window=lookback)
        volatility_factor = FactorRegistry.get("volatility", window=lookback)
        atr_pct_factor = FactorRegistry.get("atr_pct")
        
        factors = {
            'momentum': momentum_factor.compute_batch(history),
            'volatility': volatility_factor.compute_batch(history),
            'atr_pct': atr_pct_factor.compute_batch(history),
        }
        
        self.logger.info(f"Computed momentum factors for {len(history)} stocks")
        
        return factors
    
    def generate_signals(self, context: StrategyContext) -> List[Signal]:
        """生成交易信号 - 向量化实现"""
        signals = []

        top_n = self.get_param('top_n')
        min_mom = self.get_param('min_momentum')
        max_vol = self.get_param('max_volatility')
        use_vol_weight = self.get_param('use_volatility_weight')

        # 向量化收集候选 - 避免iterrows
        df = context.current_data.copy()

        # 获取因子值 (向量化)
        df['momentum'] = df['code'].apply(lambda x: context.get_factor('momentum', x))
        df['volatility'] = df['code'].apply(lambda x: context.get_factor('volatility', x))
        df['atr_pct'] = df['code'].apply(lambda x: context.get_factor('atr_pct', x))

        # 过滤条件 (向量化)
        mask = df['momentum'].notna()
        mask &= df['momentum'] >= min_mom
        mask &= (df['volatility'].isna() | (df['volatility'] <= max_vol))

        candidates_df = df[mask].copy()

        if candidates_df.empty:
            # 生成卖出信号 (清仓所有)
            for code in list(context.positions.keys()):
                signals.append(Signal(
                    code=code,
                    side=OrderSide.SELL,
                    weight=0.0,
                    reason="无符合条件的候选股票"
                ))
            return signals

        # 填充缺失值
        candidates_df['volatility'] = candidates_df['volatility'].fillna(0.3)
        candidates_df['atr_pct'] = candidates_df['atr_pct'].fillna(0.02)

        # 排序选 Top N (向量化)
        selected_df = candidates_df.nlargest(top_n, 'momentum')

        # 计算权重 (向量化)
        if use_vol_weight and not selected_df.empty:
            # 波动率倒数加权 (向量化)
            inv_vol = 1 / selected_df['volatility'].clip(lower=0.1)
            weights = (inv_vol / inv_vol.sum()).values
        else:
            weights = np.ones(len(selected_df)) / len(selected_df) if not selected_df.empty else []

        # 生成卖出信号 (向量化集合操作)
        selected_codes = set(selected_df['code'].values)
        for code in list(context.positions.keys()):
            if code not in selected_codes:
                signals.append(Signal(
                    code=code,
                    side=OrderSide.SELL,
                    weight=0.0,
                    reason="动量排名下降，清仓"
                ))

        # 生成买入信号 (向量化)
        for (_, row), w in zip(selected_df.iterrows(), weights):
            current_weight = context.positions.get(row['code'], 0) / context.total_equity if context.total_equity > 0 else 0

            # 只在权重变化较大时调整
            if abs(w - current_weight) > 0.02:
                signals.append(Signal(
                    code=row['code'],
                    side=OrderSide.BUY,
                    weight=w * 0.95,  # 保留现金
                    price=row['close'],
                    reason=f"MOM={row['momentum']:.1%} VOL={row['volatility']:.1%}"
                ))

        return signals