# ============================================================================
# 文件: strategy/hanbing_strategy.py
# ============================================================================
"""
寒冰策略 - 洗盘识别和底部确认

策略逻辑:
┌─────────────────────────────────────────────────────────────────────────────┐
│                          HanbingStrategy                                     │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  【核心理念】 识别洗盘后的底部反转机会，捕捉主力资金建仓后的突破                │
│                                                                             │
│  【洗盘识别】 ALL 必须满足:                                                  │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ 1. 价格形态:                                                          │   │
│  │    - 价格在60日高点 -20% 至 -40% 区间                                  │   │
│  │    - 最近20日波动率 < 历史均值 × 0.8 (缩量整理)                        │   │
│  │    - 成交量持续萎缩 (< 5日均量的 70%)                                  │   │
│  │                                                                         │   │
│  │ 2. 洗盘特征:                                                          │   │
│  │    - 至少出现过一次放量阴线 (跌幅 > 5%)                                 │   │
│  │    - 之后出现缩量小阴小阳震荡 (洗盘特征)                                │   │
│  │    - 振幅逐渐收窄 (< 历史振幅的 60%)                                    │   │
│  │                                                                         │   │
│  │ 3. 技术指标:                                                          │   │
│  │    - RSI 在 30-50 区间震荡 (底部特征)                                  │   │
│  │    - MACD 接近零轴或小幅红柱 (即将突破)                                 │   │
│  │    - 布林带收窄 (波动率收敛)                                           │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  【反转确认】 ALL 必须满足:                                                  │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ 1. 反转信号:                                                          │   │
│  │    - 放量阳线突破 5日或 10 日高点                                      │   │
│  │    - 或金叉信号 (MACD/RSI/RSRS)                                       │   │
│  │    - 涨幅 > 3% 且成交量 > 5日均量的 1.5倍                              │   │
│  │                                                                         │   │
│  │ 2. 趋势确认:                                                          │   │
│  │    - MA5 开始向上拐头                                                 │   │
│  │    - 价格站稳 5 日均线                                                 │   │
│  │    - 连续 2-3 根小阳线 (温和上涨)                                      │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  【离场条件】 ANY 触发:                                                      │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ 1. 止损: 价格 < 入场价 × (1 - 8%)                                     │   │
│  │ 2. 止盈: 盈利 > 25%                                                   │   │
│  │ 3. 趋势破坏: 价格 < MA20 或 跌破前低                                   │   │
│  │ 4. 成交量异常: 异常放量下跌                                             │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  【仓位管理】                                                                │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ 底部确认仓: 30%                                                       │   │
│  │ 突破加仓: 再加 20% (突破重要阻力位)                                    │   │
│  │ 趋势强化: 再加 10% (MA60上穿MA120)                                    │   │
│  │ 单只最大: 60%                                                         │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime

from .base import BaseStrategy, Signal, OrderSide, StrategyContext
from .registry import StrategyRegistry
from factors import FactorRegistry
from config import settings


class WashOutPhase(Enum):
    """洗盘阶段"""
    NONE = 0
    DIP = 1          # 下跌洗盘
    CONSOLIDATION = 2  # 震荡洗盘
    BREAKOUT = 3     # 突破确认


@dataclass
class WashOutState:
    """洗盘状态"""
    code: str
    phase: WashOutPhase = WashOutPhase.NONE
    high_price: float = 0.0          # 近期高点
    consolidation_start_date: Optional[str] = None
    volatility_ratio: float = 0.0     # 波动率比率
    volume_ratio: float = 0.0         # 成交量比率
    has_wash_signal: bool = False    # 是否有洗盘信号


@dataclass
class HanbingPosition:
    """寒冰策略持仓"""
    code: str
    entry_date: str
    entry_price: float
    quantity: int
    base_position: float = 0.0       # 基础仓位
    added_position: float = 0.0      # 加仓位
    highest_price: float = field(default=0.0)
    lowest_price: float = field(default=0.0)
    highest_lowest: float = field(default=0.0)  # 最高最低价
    breakout_count: int = 0          # 突破次数
    
    def total_weight(self) -> float:
        """总仓位"""
        return self.base_position + self.added_position
    
    def update_prices(self, current_price: float):
        """更新价格"""
        self.highest_price = max(self.highest_price, current_price)
        self.lowest_price = min(self.lowest_price, current_price) if self.lowest_price > 0 else current_price
        self.highest_lowest = max(self.highest_lowest, current_price)
    
    def get_pnl_pct(self, current_price: float) -> float:
        """获取盈亏百分比"""
        return (current_price - self.entry_price) / self.entry_price


@StrategyRegistry.register
class HanbingStrategy(BaseStrategy):
    """
    寒冰策略
    
    特点:
    - 识别洗盘后的底部反转
    - 低风险抄底
    - 趋势确认加仓
    - 成交量分析
    """
    
    name = "hanbing"
    version = "1.0.0"
    
    def __init__(self, params: Dict = None):
        default_params = {
            # 洗盘识别参数
            'dip_range_min': -0.40,        # 最小跌幅 40%
            'dip_range_max': -0.20,        # 最大跌幅 20%
            'low_vol_ratio': 0.70,         # 低成交量比率
            'volatility_contraction': 0.8,  # 波动率收缩系数
            'wash_down_threshold': 0.05,   # 洗盘下跌阈值 5%
            'range_contraction': 0.6,      # 振幅收缩系数
            'rsi_range': (30, 50),        # RSI范围
            
            # 反转确认参数
            'breakout_gain': 0.03,        # 突破涨幅 3%
            'breakout_volume_ratio': 1.5,  # 突破成交量比
            'ma5_slope_threshold': 0.001, # MA5斜率阈值
            'consecutive_bars': 2,         # 连续阳线根数
            
            # 出场参数
            'stop_loss_pct': 0.08,        # 止损 8%
            'take_profit_pct': 0.25,       # 止盈 25%
            'trend_break_threshold': -0.02, # 趋势破坏阈值 -2%
            
            # 仓位参数
            'base_position_pct': 0.30,    # 基础仓位 30%
            'add_breakout_pct': 0.20,      # 突破加仓 20%
            'add_trend_pct': 0.10,        # 趋势加仓 10%
            'max_single_position': 0.60,  # 单只最大仓位 60%
        }
        
        merged_params = {**default_params, **(params or {})}
        super().__init__(merged_params)
        
        # 持仓管理
        self._positions: Dict[str, HanbingPosition] = {}
        
        # 洗盘状态跟踪
        self._washout_states: Dict[str, WashOutState] = {}
    
    def compute_factors(self, history: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """计算因子"""
        # 获取因子
        rsrs_factor = FactorRegistry.get("rsrs_slope")
        momentum_factor = FactorRegistry.get("momentum")
        atr_factor = FactorRegistry.get("atr_pct")
        volatility_factor = FactorRegistry.get("volatility")
        
        factors = {
            'rsrs': rsrs_factor.compute_batch(history),
            'momentum': momentum_factor.compute_batch(history),
            'atr_pct': atr_factor.compute_batch(history),
            'volatility': volatility_factor.compute_batch(history),
        }
        
        # 手动计算技术指标和洗盘特征因子
        technical_factors = self._compute_technical_indicators(history)
        factors.update(technical_factors)
        
        self.logger.info(f"Computed hanbing factors for {len(history)} stocks")
        
        return factors
    
    def _compute_technical_indicators(self, history: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """计算技术指标"""
        ma5 = 5
        ma10 = 10
        ma20 = 20
        ma60 = 60
        
        ma5_dict = {}
        ma10_dict = {}
        ma20_dict = {}
        ma60_dict = {}
        highest_60_dict = {}
        washout_signal_dict = {}
        range_ratio_dict = {}
        
        for code, df in history.items():
            if len(df) < ma60:
                continue
            
            df = df.copy()
            
            # 计算MA
            df['ma5'] = df['close'].rolling(ma5, min_periods=1).mean()
            df['ma10'] = df['close'].rolling(ma10, min_periods=1).mean()
            df['ma20'] = df['close'].rolling(ma20, min_periods=1).mean()
            df['ma60'] = df['close'].rolling(ma60, min_periods=1).mean()
            
            # 计算最高价
            df['highest_60'] = df['high'].rolling(ma60, min_periods=1).max()
            
            # 距高点跌幅
            df['dip_from_high'] = (df['close'] - df['highest_60']) / df['highest_60']
            
            # 洗盘信号: 放量阴线
            df['avg_volume'] = df['vol'].rolling(20, min_periods=1).mean()
            df['daily_change'] = df['close'].pct_change()
            df['wash_signal'] = (
                (df['daily_change'] < -self.get_param('wash_down_threshold')) &
                (df['vol'] > df['avg_volume'] * 2)
            ).astype(float)
            
            # 振幅
            df['range'] = (df['high'] - df['low']) / df['close']
            df['avg_range'] = df['range'].rolling(20, min_periods=1).mean()
            df['range_ratio'] = df['range'] / df['avg_range']
            
            ma5_dict[code] = df['ma5']
            ma10_dict[code] = df['ma10']
            ma20_dict[code] = df['ma20']
            ma60_dict[code] = df['ma60']
            highest_60_dict[code] = df['highest_60']
            washout_signal_dict[code] = df['wash_signal']
            range_ratio_dict[code] = df['range_ratio']
        
        # 转换为DataFrame
        all_dates = set()
        for s in ma5_dict.values():
            all_dates.update(s.index)
        for s in ma10_dict.values():
            all_dates.update(s.index)
        for s in ma20_dict.values():
            all_dates.update(s.index)
        for s in ma60_dict.values():
            all_dates.update(s.index)
        for s in highest_60_dict.values():
            all_dates.update(s.index)
        for s in washout_signal_dict.values():
            all_dates.update(s.index)
        for s in range_ratio_dict.values():
            all_dates.update(s.index)
        
        all_dates = sorted(list(all_dates))
        
        ma5_df = pd.DataFrame(index=all_dates)
        ma10_df = pd.DataFrame(index=all_dates)
        ma20_df = pd.DataFrame(index=all_dates)
        ma60_df = pd.DataFrame(index=all_dates)
        highest_60_df = pd.DataFrame(index=all_dates)
        washout_signal_df = pd.DataFrame(index=all_dates)
        range_ratio_df = pd.DataFrame(index=all_dates)
        
        for code, s in ma5_dict.items():
            ma5_df[code] = s
        
        for code, s in ma10_dict.items():
            ma10_df[code] = s
        
        for code, s in ma20_dict.items():
            ma20_df[code] = s
        
        for code, s in ma60_dict.items():
            ma60_df[code] = s
        
        for code, s in highest_60_dict.items():
            highest_60_df[code] = s
        
        for code, s in washout_signal_dict.items():
            washout_signal_df[code] = s
        
        for code, s in range_ratio_dict.items():
            range_ratio_df[code] = s
        
        return {
            'ma5': ma5_df,
            'ma10': ma10_df,
            'ma20': ma20_df,
            'ma60': ma60_df,
            'highest_60': highest_60_df,
            'wash_signal': washout_signal_df,
            'range_ratio': range_ratio_df,
        }
    
    def generate_signals(self, context: StrategyContext) -> List[Signal]:
        """生成交易信号"""
        signals = []
        
        df = context.current_data.copy()
        
        # 获取因子值
        df['ma5'] = df['code'].apply(lambda x: context.get_factor('ma5', x))
        df['ma10'] = df['code'].apply(lambda x: context.get_factor('ma10', x))
        df['ma20'] = df['code'].apply(lambda x: context.get_factor('ma20', x))
        df['ma60'] = df['code'].apply(lambda x: context.get_factor('ma60', x))
        df['volatility'] = df['code'].apply(lambda x: context.get_factor('volatility', x))
        df['rsrs'] = df['code'].apply(lambda x: context.get_factor('rsrs', x))
        df['highest_60'] = df['code'].apply(lambda x: context.get_factor('highest_60', x))
        df['wash_signal'] = df['code'].apply(lambda x: context.get_factor('wash_signal', x))
        df['range_ratio'] = df['code'].apply(lambda x: context.get_factor('range_ratio', x))
        df['momentum'] = df['code'].apply(lambda x: context.get_factor('momentum', x))
        df['atr_pct'] = df['code'].apply(lambda x: context.get_factor('atr_pct', x))
        
        # 计算距离高点跌幅
        df['dip_from_high'] = (df['close'] - df['highest_60']) / df['highest_60']
        
        # 过滤有效数据
        valid_mask = (
            df['ma5'].notna() &
            df['ma20'].notna() &
            df['ma60'].notna() &
            df['volatility'].notna()
        )
        
        df = df[valid_mask].copy()
        
        if df.empty:
            return signals
        
        # ========== 更新洗盘状态 ==========
        self._update_washout_states(df, context.current_date)
        
        # ========== 检查现有持仓 ==========
        for code in list(self._positions.keys()):
            pos = self._positions[code]
            
            if code not in df['code'].values:
                # 平仓
                signals.append(Signal(
                    code=code,
                    side=OrderSide.SELL,
                    weight=0.0,
                    reason="股票不在交易列表"
                ))
                del self._positions[code]
                continue
            
            row = df[df['code'] == code].iloc[0]
            current_price = row['close']
            daily_gain = row['daily_gain']
            
            pos.update_prices(current_price)
            
            # 检查离场条件
            exit_signal = self._check_exit(pos, current_price, daily_gain, row)
            if exit_signal:
                signals.append(exit_signal)
                del self._positions[code]
        
        # ========== 生成新信号 ==========
        
        # 筛选洗盘状态为 BREAKOUT 的股票
        breakout_candidates = []
        for code, state in self._washout_states.items():
            if state.phase == WashOutState.BREAKOUT:
                if code in df['code'].values:
                    row = df[df['code'] == code].iloc[0]
                    breakout_candidates.append(row)
        
        if breakout_candidates:
            breakout_df = pd.DataFrame(breakout_candidates)
            
            # 进一步筛选反转确认条件
            reversal_mask = (
                (breakout_df['momentum'] >= self.get_param('breakout_gain')) &
                (breakout_df['close'] > breakout_df['ma5']) &
                (breakout_df['rsrs'] >= 0.5)
            )
            
            reversal_stocks = breakout_df[reversal_mask]
            
            if not reversal_stocks.empty:
                # 按动量排序选择
                top_reversals = reversal_stocks.nlargest(5, 'momentum')
                
                for _, row in top_reversals.iterrows():
                    code = row['code']
                    
                    if code in self._positions:
                        continue
                    
                    # 新建仓
                    base_position = self.get_param('base_position_pct')
                    self._positions[code] = HanbingPosition(
                        code=code,
                        entry_date=context.current_date,
                        entry_price=row['close'],
                        quantity=0,
                        base_position=base_position,
                        highest_price=row['close'],
                        lowest_price=row['close'],
                        breakout_count=1
                    )
                    
                    signals.append(Signal(
                        code=code,
                        side=OrderSide.BUY,
                        weight=base_position,
                        price=row['close'],
                        reason=f"洗盘反转 动量={row['momentum']:.2%} RSRS={row['rsrs']:.2f}"
                    ))
        
        return signals
    
    def _update_washout_states(self, df: pd.DataFrame, current_date: str):
        """更新洗盘状态"""
        dip_range_min = self.get_param('dip_range_min')
        dip_range_max = self.get_param('dip_range_max')
        low_vol_ratio = self.get_param('low_vol_ratio')
        
        for code in df['code'].unique():
            row = df[df['code'] == code].iloc[0]
            
            # 初始化状态
            if code not in self._washout_states:
                self._washout_states[code] = WashOutState(code=code)
            
            state = self._washout_states[code]
            
            # 判断洗盘阶段
            if dip_range_min <= row['dip_from_high'] <= dip_range_max:
                # 在跌幅范围内
                if row['volume_ratio_5'] <= low_vol_ratio:
                    # 缩量整理
                    if state.phase == WashOutPhase.NONE:
                        state.phase = WashOutPhase.DIP
                    elif state.phase == WashOutPhase.DIP:
                        state.phase = WashOutPhase.CONSOLIDATION
                        state.consolidation_start_date = current_date
            elif row['daily_gain'] >= self.get_param('breakout_gain'):
                # 突破确认
                if state.phase == WashOutPhase.CONSOLIDATION:
                    state.phase = WashOutPhase.BREAKOUT
            else:
                # 重置状态
                state.phase = WashOutPhase.NONE
                state.consolidation_start_date = None
    
    def _check_exit(self, pos: HanbingPosition, current_price: float,
                     daily_gain: float, row: pd.Series) -> Optional[Signal]:
        """检查是否需要离场"""
        stop_loss_pct = self.get_param('stop_loss_pct')
        take_profit_pct = self.get_param('take_profit_pct')
        trend_break_threshold = self.get_param('trend_break_threshold')
        
        reason = None
        
        # 计算盈亏
        pnl_pct = pos.get_pnl_pct(current_price)
        
        # 止损
        if pnl_pct <= -stop_loss_pct:
            reason = f"止损 亏损={pnl_pct:.2%}"
        # 止盈
        elif pnl_pct >= take_profit_pct:
            reason = f"止盈 盈利={pnl_pct:.2%}"
        # 趋势破坏
        elif daily_gain <= trend_break_threshold or current_price < row['ma20']:
            reason = f"趋势破坏 涨幅={daily_gain:.2%}"
        
        if reason:
            return Signal(
                code=pos.code,
                side=OrderSide.SELL,
                weight=0.0,
                reason=reason
            )
        
        return None
    
    def on_order_filled(self, order: 'Order') -> None:
        """订单成交回调"""
        pass
    
    def on_day_end(self, context: StrategyContext) -> None:
        """日终回调"""
        # 清理过期的洗盘状态
        for code, state in list(self._washout_states.items()):
            if state.consolidation_start_date:
                # 洗盘时间超过60天则重置
                start_date = pd.Timestamp(state.consolidation_start_date)
                current_date = pd.Timestamp(context.current_date)
                if (current_date - start_date).days > 60:
                    state.phase = WashOutPhase.NONE
                    state.consolidation_start_date = None
