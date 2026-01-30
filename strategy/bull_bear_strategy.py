# ============================================================================
# 文件: strategy/bull_bear_strategy.py
# ============================================================================
"""
牛熊策略 - 基于移动平均线的多空判断

策略逻辑:
┌─────────────────────────────────────────────────────────────────────────────┐
│                           BullBearStrategy                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  【多头信号】 ALL 必须满足:                                                   │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ 1. 价格突破近期高点 (N日最高价)                                      │   │
│  │ 2. MA5 > MA20 > MA60 (多头排列)                                    │   │
│  │ 3. RSRS > 0.5 (相对强度确认)                                       │   │
│  │ 4. 成交量放大 (Volume > MA5_Vol × 1.2)                              │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  【空头信号】 ALL 必须满足:                                                   │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ 1. 价格跌破近期低点 (N日最低价)                                      │   │
│  │ 2. MA5 < MA20 < MA60 (空头排列)                                    │   │
│  │ 3. RSRS < -0.5 (相对强度确认)                                      │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  【止损机制】                                                                │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ 1. 多头止损: 价格 < 入场价 × (1 - 止损比例)                          │   │
│  │ 2. 空头止损: 价格 > 入场价 × (1 + 止损比例)                          │   │
│  │ 3. 趋势反转: MA5与MA20交叉                                           │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  【仓位管理】                                                                │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ 基础仓位: 总资金的 30%                                              │   │
│  │ 加仓条件: 盈利 > 10% 且趋势强化                                     │   │
│  │ 最大仓位: 单只 40%                                                   │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from dataclasses import dataclass, field
from datetime import datetime, timedelta

from .base import BaseStrategy, Signal, OrderSide, StrategyContext
from .registry import StrategyRegistry
from factors import FactorRegistry
from config import settings


@dataclass
class BullBearPosition:
    """牛熊策略持仓状态"""
    code: str
    side: OrderSide
    entry_price: float
    entry_date: str
    quantity: int
    highest_price: float = field(default=0.0)
    lowest_price: float = field(default=0.0)
    
    def update_prices(self, price: float):
        """更新最高/最低价"""
        if self.side == OrderSide.BUY:
            self.highest_price = max(self.highest_price, price)
        else:
            self.lowest_price = min(self.lowest_price, price)
    
    def get_pnl_pct(self, current_price: float) -> float:
        """获取盈亏百分比"""
        if self.side == OrderSide.BUY:
            return (current_price - self.entry_price) / self.entry_price
        else:
            return (self.entry_price - current_price) / self.entry_price


@StrategyRegistry.register
class BullBearStrategy(BaseStrategy):
    """
    牛熊策略
    
    特点:
    - 趋势跟随，双向交易
    - 移动平均线判断多空
    - 动态止损止盈
    - 仓位动态调整
    """
    
    name = "bull_bear"
    version = "1.0.0"
    
    def __init__(self, params: Dict = None):
        default_params = {
            # 入场参数
            'lookback_high': 20,            # 突破高点回溯天数
            'lookback_low': 20,             # 突破低点回溯天数
            'ma_short': 5,                  # 短期MA
            'ma_medium': 20,                # 中期MA
            'ma_long': 60,                  # 长期MA
            'volume_multiplier': 1.2,       # 放量倍数
            
            # 出场参数
            'stop_loss_pct': 0.05,          # 止损比例 5%
            'take_profit_pct': 0.15,        # 止盈比例 15%
            'trend_reversal': True,         # 是否使用趋势反转止损
            
            # 仓位参数
            'base_position_pct': 0.30,      # 基础仓位 30%
            'max_position_pct': 0.40,      # 最大仓位 40%
            'add_position_pct': 0.10,      # 加仓比例 10%
            'add_profit_threshold': 0.10,   # 加仓盈利阈值 10%
        }
        
        merged_params = {**default_params, **(params or {})}
        super().__init__(merged_params)
        
        # 持仓状态
        self._positions: Dict[str, BullBearPosition] = {}
    
    def compute_factors(self, history: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """计算因子"""
        ma_short = self.get_param('ma_short')
        ma_medium = self.get_param('ma_medium')
        ma_long = self.get_param('ma_long')
        
        # 获取因子
        rsrs_factor = FactorRegistry.get("rsrs_slope")
        momentum_factor = FactorRegistry.get("momentum")
        
        factors = {
            'rsrs': rsrs_factor.compute_batch(history),
            'momentum': momentum_factor.compute_batch(history),
        }
        
        # 手动计算MA和突破因子
        technical_factors = self._compute_technical_indicators(history, ma_short, ma_medium, ma_long)
        factors.update(technical_factors)
        
        self.logger.info(f"Computed bull_bear factors for {len(history)} stocks")
        
        return factors
    
    def _compute_technical_indicators(self, history: Dict[str, pd.DataFrame],
                                      ma_short: int, ma_medium: int, ma_long: int) -> Dict[str, pd.DataFrame]:
        """计算技术指标"""
        lookback_high = self.get_param('lookback_high')
        lookback_low = self.get_param('lookback_low')
        
        ma5_dict = {}
        ma20_dict = {}
        ma60_dict = {}
        dist_to_high_dict = {}
        dist_to_low_dict = {}
        
        for code, df in history.items():
            if len(df) < max(ma_long, lookback_high, lookback_low):
                continue
            
            df = df.copy()
            
            # 计算MA
            df['ma5'] = df['close'].rolling(ma_short, min_periods=1).mean()
            df['ma20'] = df['close'].rolling(ma_medium, min_periods=1).mean()
            df['ma60'] = df['close'].rolling(ma_long, min_periods=1).mean()
            
            # 计算N日最高价和最低价
            df['high_n'] = df['high'].rolling(lookback_high, min_periods=1).max()
            df['low_n'] = df['low'].rolling(lookback_low, min_periods=1).min()
            
            # 高点距离
            df['dist_to_high'] = (df['close'] - df['high_n']) / df['high_n']
            # 低点距离
            df['dist_to_low'] = (df['low_n'] - df['close']) / df['low_n']
            
            ma5_dict[code] = df['ma5']
            ma20_dict[code] = df['ma20']
            ma60_dict[code] = df['ma60']
            dist_to_high_dict[code] = df['dist_to_high']
            dist_to_low_dict[code] = df['dist_to_low']
        
        # 转换为DataFrame格式
        all_dates = set()
        for s in ma5_dict.values():
            all_dates.update(s.index)
        for s in ma20_dict.values():
            all_dates.update(s.index)
        for s in ma60_dict.values():
            all_dates.update(s.index)
        for s in dist_to_high_dict.values():
            all_dates.update(s.index)
        for s in dist_to_low_dict.values():
            all_dates.update(s.index)
        
        all_dates = sorted(list(all_dates))
        
        ma5_df = pd.DataFrame(index=all_dates)
        ma20_df = pd.DataFrame(index=all_dates)
        ma60_df = pd.DataFrame(index=all_dates)
        dist_to_high_df = pd.DataFrame(index=all_dates)
        dist_to_low_df = pd.DataFrame(index=all_dates)
        
        for code, s in ma5_dict.items():
            ma5_df[code] = s
        
        for code, s in ma20_dict.items():
            ma20_df[code] = s
        
        for code, s in ma60_dict.items():
            ma60_df[code] = s
        
        for code, s in dist_to_high_dict.items():
            dist_to_high_df[code] = s
        
        for code, s in dist_to_low_dict.items():
            dist_to_low_df[code] = s
        
        return {
            'ma5': ma5_df,
            'ma20': ma20_df,
            'ma60': ma60_df,
            'dist_to_high': dist_to_high_df,
            'dist_to_low': dist_to_low_df,
        }
    
    def generate_signals(self, context: StrategyContext) -> List[Signal]:
        """生成交易信号"""
        signals = []
        
        ma_short = self.get_param('ma_short')
        ma_medium = self.get_param('ma_medium')
        ma_long = self.get_param('ma_long')
        volume_multiplier = self.get_param('volume_multiplier')
        
        df = context.current_data.copy()
        
        # 获取因子值 (向量化)
        df['ma5'] = df['code'].apply(lambda x: context.get_factor('ma5', x))
        df['ma20'] = df['code'].apply(lambda x: context.get_factor('ma20', x))
        df['ma60'] = df['code'].apply(lambda x: context.get_factor('ma60', x))
        df['rsrs'] = df['code'].apply(lambda x: context.get_factor('rsrs', x))
        df['dist_to_high'] = df['code'].apply(lambda x: context.get_factor('dist_to_high', x))
        df['dist_to_low'] = df['code'].apply(lambda x: context.get_factor('dist_to_low', x))
        df['momentum'] = df['code'].apply(lambda x: context.get_factor('momentum', x))
        
        # 过滤有效数据
        valid_mask = (
            df['ma5'].notna() &
            df['ma20'].notna() &
            df['ma60'].notna() &
            df['rsrs'].notna()
        )
        
        df = df[valid_mask].copy()
        
        if df.empty:
            return signals
        
        # ========== 检查现有持仓 ==========
        for code in list(self._positions.keys()):
            pos = self._positions[code]
            
            if code not in df['code'].values:
                # 股票不在交易列表，平仓
                signals.append(self._create_exit_signal(pos, "股票不在交易列表"))
                del self._positions[code]
                continue
            
            row = df[df['code'] == code].iloc[0]
            current_price = row['close']
            
            # 更新持仓价格
            pos.update_prices(current_price)
            
            # 检查止损止盈
            exit_signal = self._check_exit(pos, current_price, row)
            if exit_signal:
                signals.append(exit_signal)
                del self._positions[code]
                continue
        
        # ========== 生成新信号 ==========
        
        # 多头信号筛选
        bull_mask = (
            (df['dist_to_high'] >= 0) &  # 突破高点
            (df['ma5'] > df['ma20']) &  # MA5 > MA20
            (df['ma20'] > df['ma60']) &  # MA20 > MA60
            (df['rsrs'] > 0.5) &  # RSRS > 0.5
            (df['vol'] > df['vol'].mean() * volume_multiplier)  # 放量
        )
        
        # 空头信号筛选
        bear_mask = (
            (df['dist_to_low'] >= 0) &  # 跌破低点
            (df['ma5'] < df['ma20']) &  # MA5 < MA20
            (df['ma20'] < df['ma60']) &  # MA20 < MA60
            (df['rsrs'] < -0.5)  # RSRS < -0.5
        )
        
        bull_candidates = df[bull_mask]
        bear_candidates = df[bear_mask]
        
        # 生成多头信号 (选择突破力度最大的)
        if not bull_candidates.empty:
            # 按RSRS排序选择最优
            top_bull = bull_candidates.nlargest(5, 'rsrs')
            for _, row in top_bull.iterrows():
                code = row['code']
                if code not in self._positions:
                    weight = self.get_param('base_position_pct')
                    signals.append(Signal(
                        code=code,
                        side=OrderSide.BUY,
                        weight=weight,
                        price=row['close'],
                        reason=f"突破高点 RSRS={row['rsrs']:.2f}"
                    ))
        
        # 生成空头信号 (选择RSRS最低的)
        if not bear_candidates.empty:
            # 按RSRS排序选择最优
            top_bear = bear_candidates.nsmallest(5, 'rsrs')
            for _, row in top_bear.iterrows():
                code = row['code']
                if code not in self._positions:
                    weight = self.get_param('base_position_pct')
                    signals.append(Signal(
                        code=code,
                        side=OrderSide.SELL,
                        weight=weight,
                        price=row['close'],
                        reason=f"跌破低点 RSRS={row['rsrs']:.2f}"
                    ))
        
        return signals
    
    def _check_exit(self, pos: BullBearPosition, current_price: float, 
                     row: pd.Series) -> Optional[Signal]:
        """检查是否需要离场"""
        stop_loss_pct = self.get_param('stop_loss_pct')
        take_profit_pct = self.get_param('take_profit_pct')
        use_trend_reversal = self.get_param('trend_reversal')
        
        # 计算盈亏
        pnl_pct = pos.get_pnl_pct(current_price)
        
        reason = None
        
        if pos.side == OrderSide.BUY:
            # 多头止损
            if pnl_pct <= -stop_loss_pct:
                reason = f"多头止损 亏损={pnl_pct:.2%}"
            # 多头止盈
            elif pnl_pct >= take_profit_pct:
                reason = f"多头止盈 盈利={pnl_pct:.2%}"
            # 趋势反转
            elif use_trend_reversal:
                if row['ma5'] < row['ma20']:
                    reason = "趋势反转 MA5下穿MA20"
        else:
            # 空头止损
            if pnl_pct <= -stop_loss_pct:
                reason = f"空头止损 亏损={pnl_pct:.2%}"
            # 空头止盈
            elif pnl_pct >= take_profit_pct:
                reason = f"空头止盈 盈利={pnl_pct:.2%}"
            # 趋势反转
            elif use_trend_reversal:
                if row['ma5'] > row['ma20']:
                    reason = "趋势反转 MA5上穿MA20"
        
        if reason:
            return self._create_exit_signal(pos, reason)
        
        return None
    
    def _create_exit_signal(self, pos: BullBearPosition, reason: str) -> Signal:
        """创建离场信号"""
        return Signal(
            code=pos.code,
            side=pos.side,  # 平仓信号与持仓方向相同，由引擎处理平仓
            weight=0.0,
            reason=reason
        )
    
    def on_order_filled(self, order: 'Order') -> None:
        """订单成交回调 - 更新持仓状态"""
        # 这里可以添加持仓状态更新逻辑
        pass
    
    def on_day_end(self, context: StrategyContext) -> None:
        """日终回调 - 检查加仓"""
        add_profit_threshold = self.get_param('add_profit_threshold')
        add_position_pct = self.get_param('add_position_pct')
        max_position_pct = self.get_param('max_position_pct')
        
        for code, pos in self._positions.items():
            # 获取当前价格
            if code not in context.current_data['code'].values:
                continue
            
            row = context.current_data[context.current_data['code'] == code].iloc[0]
            current_price = row['close']
            pnl_pct = pos.get_pnl_pct(current_price)
            
            # 检查是否满足加仓条件
            if pnl_pct >= add_profit_threshold and pos.side == OrderSide.BUY:
                # 这里可以添加加仓逻辑
                # 需要发送额外的买入信号
                pass
