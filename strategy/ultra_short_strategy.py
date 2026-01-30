# ============================================================================
# 文件: strategy/ultra_short_strategy.py
# ============================================================================
"""
超短线策略 - 1-5分钟K线高频交易

策略逻辑:
┌─────────────────────────────────────────────────────────────────────────────┐
│                         UltraShortStrategy                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  【适用场景】 1-5分钟级别K线，日内或隔夜短线                                 │
│                                                                             │
│  【入场条件】 ALL 必须满足:                                                  │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ 1. 快速突破: 价格突破最近3根K线的高点/低点                          │   │
│  │ 2. 动量确认: 3分钟内涨跌幅 > 1.5%                                   │   │
│  │ 3. 成交量确认: Volume > 均值 × 1.8                                  │   │
│  │ 4. 价格远离MA: |Price - MA5| / Price > 0.5% (避免震荡市)            │   │
│  │ 5. 波动率合适: ATR/Price < 2% (避免极端波动)                       │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  【离场条件】 ANY 触发:                                                      │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ 1. 快速止损: 亏损 > 2%                                               │   │
│  │ 2. 时间止损: 持仓 > 10根K线                                           │   │
│  │ 3. 动能衰减: 涨跌幅 < 0.3% 且成交萎缩                                 │   │
│  │ 4. 快速止盈: 盈利 > 3%                                               │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  【仓位管理】                                                                │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ 基础仓位: 单笔 20%                                                  │   │
│  │ 加仓: 连续盈利 > 3笔后加仓至 30%                                    │   │
│  │ 同向持仓限制: 最多 3个同向仓位                                      │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  【风险控制】                                                                │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ 单日最大亏损: 5%                                                     │   │
│  │ 单日最大交易次数: 20次                                                │   │
│  │ 连续亏损限制: 连续3次亏损后暂停1小时                                  │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import deque

from .base import BaseStrategy, Signal, OrderSide, StrategyContext
from .registry import StrategyRegistry
from factors import FactorRegistry
from config import settings


@dataclass
class TradeRecord:
    """交易记录"""
    timestamp: datetime
    code: str
    side: OrderSide
    entry_price: float
    exit_price: float
    pnl: float
    pnl_pct: float
    bars_held: int


@dataclass
class UltraShortPosition:
    """超短线持仓"""
    code: str
    side: OrderSide
    entry_price: float
    entry_time: datetime
    entry_bar_index: int
    quantity: int
    initial_momentum: float
    bars_held: int = 0
    
    def update(self, current_price: float, momentum: float) -> Tuple[bool, Optional[str]]:
        """
        更新持仓，检查是否需要离场
        
        Returns:
            (should_exit, reason)
        """
        self.bars_held += 1
        
        # 计算盈亏
        if self.side == OrderSide.BUY:
            pnl_pct = (current_price - self.entry_price) / self.entry_price
        else:
            pnl_pct = (self.entry_price - current_price) / self.entry_price
        
        # 检查止损止盈条件
        # 快速止损
        if pnl_pct <= -0.02:  # -2%
            return True, f"快速止损 {pnl_pct:.2%}"
        
        # 快速止盈
        if pnl_pct >= 0.03:  # +3%
            return True, f"快速止盈 {pnl_pct:.2%}"
        
        # 时间止损
        if self.bars_held >= 10:
            return True, f"时间止损 持仓{self.bars_held}根K线"
        
        # 动能衰减
        if abs(momentum) < 0.003 and self.bars_held >= 3:
            return True, f"动能衰减 动量={momentum:.2%}"
        
        return False, None


@StrategyRegistry.register
class UltraShortStrategy(BaseStrategy):
    """
    超短线策略
    
    特点:
    - 1-5分钟级别高频交易
    - 快进快出，严格控制风险
    - 动量驱动，突破交易
    - 时间敏感，持仓时间短
    """
    
    name = "ultra_short"
    version = "1.0.0"
    
    def __init__(self, params: Dict = None):
        default_params = {
            # 入场参数
            'breakout_bars': 3,              # 突破K线数量
            'momentum_threshold': 0.015,     # 动量阈值 1.5%
            'volume_multiplier': 1.8,        # 放量倍数
            'ma_distance_threshold': 0.005, # MA距离阈值 0.5%
            'max_atr_ratio': 0.02,          # 最大ATR比例 2%
            
            # 出场参数
            'stop_loss_pct': 0.02,          # 止损 2%
            'take_profit_pct': 0.03,        # 止盈 3%
            'max_holding_bars': 10,         # 最大持仓K线数
            'momentum_decay_threshold': 0.003,  # 动能衰减阈值 0.3%
            
            # 仓位参数
            'base_position_pct': 0.20,      # 基础仓位 20%
            'add_position_pct': 0.30,       # 加仓仓位 30%
            'max_concurrent': 3,            # 最大并发持仓数
            'consecutive_wins_for_add': 3,  # 连续盈利次数后加仓
            
            # 风控参数
            'daily_max_loss_pct': 0.05,    # 单日最大亏损 5%
            'daily_max_trades': 20,         # 单日最大交易次数
            'max_consecutive_losses': 3,    # 最大连续亏损次数
            'pause_after_losses_hours': 1, # 连续亏损后暂停小时数
        }
        
        merged_params = {**default_params, **(params or {})}
        super().__init__(merged_params)
        
        # 持仓管理
        self._positions: Dict[str, UltraShortPosition] = {}
        
        # 交易统计
        self._trade_records: List[TradeRecord] = []
        self._daily_trades = 0
        self._consecutive_losses = 0
        self._last_loss_time: Optional[datetime] = None
        self._is_paused = False
        self._pause_end_time: Optional[datetime] = None
        self._current_bar_index = 0
        self._daily_pnl = 0.0
        self._daily_equity_start = 0.0
    
    def compute_factors(self, history: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """计算因子"""
        # 获取因子
        atr_factor = FactorRegistry.get("atr_pct")
        momentum_factor = FactorRegistry.get("momentum")
        
        factors = {
            'atr_pct': atr_factor.compute_batch(history),
            'momentum': momentum_factor.compute_batch(history),
        }
        
        # 手动计算技术指标和突破因子
        technical_factors = self._compute_technical_indicators(history)
        factors.update(technical_factors)
        
        self.logger.info(f"Computed ultra_short factors for {len(history)} stocks")
        
        return factors
    
    def _compute_technical_indicators(self, history: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """计算技术指标"""
        breakout_bars = self.get_param('breakout_bars')
        ma_short = 5
        
        ma5_dict = {}
        high_breakout_dict = {}
        low_breakout_dict = {}
        
        for code, df in history.items():
            if len(df) < breakout_bars + 1:
                continue
            
            df = df.copy()
            
            # 计算MA5
            df['ma5'] = df['close'].rolling(ma_short, min_periods=1).mean()
            
            # 最近N根K线的高低点
            df['rolling_high'] = df['high'].rolling(breakout_bars, min_periods=1).max().shift(1)
            df['rolling_low'] = df['low'].rolling(breakout_bars, min_periods=1).min().shift(1)
            
            # 突破信号
            df['high_breakout'] = (df['close'] > df['rolling_high']).astype(float)
            df['low_breakout'] = (df['close'] < df['rolling_low']).astype(float)
            
            ma5_dict[code] = df['ma5']
            high_breakout_dict[code] = df['high_breakout']
            low_breakout_dict[code] = df['low_breakout']
        
        # 转换为DataFrame
        all_dates = set()
        for s in ma5_dict.values():
            all_dates.update(s.index)
        for s in high_breakout_dict.values():
            all_dates.update(s.index)
        for s in low_breakout_dict.values():
            all_dates.update(s.index)
        
        all_dates = sorted(list(all_dates))
        
        ma5_df = pd.DataFrame(index=all_dates)
        high_breakout_df = pd.DataFrame(index=all_dates)
        low_breakout_df = pd.DataFrame(index=all_dates)
        
        for code, s in ma5_dict.items():
            ma5_df[code] = s
        
        for code, s in high_breakout_dict.items():
            high_breakout_df[code] = s
        
        for code, s in low_breakout_dict.items():
            low_breakout_df[code] = s
        
        return {
            'ma5': ma5_df,
            'high_breakout': high_breakout_df,
            'low_breakout': low_breakout_df,
        }
    
    def generate_signals(self, context: StrategyContext) -> List[Signal]:
        """生成交易信号"""
        signals = []
        self._current_bar_index += 1
        
        # 检查是否暂停
        if self._is_paused:
            if self._pause_end_time and datetime.now() >= self._pause_end_time:
                self._is_paused = False
                self.logger.info("恢复交易 (暂停期结束)")
            else:
                self.logger.debug("暂停交易中...")
                return signals
        
        # 检查单日最大亏损
        daily_max_loss = self.get_param('daily_max_loss_pct')
        if self._daily_equity_start > 0:
            daily_loss_pct = -self._daily_pnl / self._daily_equity_start
            if daily_loss_pct >= daily_max_loss:
                self.logger.warning(f"达到单日最大亏损 {daily_loss_pct:.2%}，暂停交易")
                return signals
        
        # 检查单日最大交易次数
        daily_max_trades = self.get_param('daily_max_trades')
        if self._daily_trades >= daily_max_trades:
            self.logger.debug(f"达到单日最大交易次数 {daily_max_trades}")
            return signals
        
        df = context.current_data.copy()
        
        # 获取因子值
        df['ma5'] = df['code'].apply(lambda x: context.get_factor('ma5', x))
        df['atr_pct'] = df['code'].apply(lambda x: context.get_factor('atr_pct', x))
        df['momentum'] = df['code'].apply(lambda x: context.get_factor('momentum', x))
        df['high_breakout'] = df['code'].apply(lambda x: context.get_factor('high_breakout', x))
        df['low_breakout'] = df['code'].apply(lambda x: context.get_factor('low_breakout', x))
        
        # 过滤有效数据
        valid_mask = (
            df['ma5'].notna() &
            df['atr_pct'].notna() &
            df['momentum'].notna()
        )
        
        df = df[valid_mask].copy()
        
        if df.empty:
            return signals
        
        # ========== 检查现有持仓 ==========
        for code in list(self._positions.keys()):
            pos = self._positions[code]
            
            if code not in df['code'].values:
                # 平仓
                exit_signal = self._create_exit_signal(pos, df[df['code'] == pos.code].iloc[0]['close'], "股票不在交易列表")
                if exit_signal:
                    signals.append(exit_signal)
                del self._positions[code]
                continue
            
            row = df[df['code'] == code].iloc[0]
            current_price = row['close']
            momentum = row['momentum']
            
            # 检查离场条件
            should_exit, reason = pos.update(current_price, momentum)
            if should_exit:
                exit_signal = self._create_exit_signal(pos, current_price, reason)
                if exit_signal:
                    signals.append(exit_signal)
                del self._positions[code]
        
        # ========== 生成新信号 ==========
        
        # 检查最大持仓数
        max_concurrent = self.get_param('max_concurrent')
        if len(self._positions) >= max_concurrent:
            return signals
        
        # 计算基础仓位
        base_position_pct = self.get_param('base_position_pct')
        if self._consecutive_losses >= self.get_param('consecutive_wins_for_add'):
            base_position_pct = self.get_param('add_position_pct')
        
        # 买入信号筛选
        buy_mask = (
            (df['high_breakout'] > 0) &  # 高点突破
            (df['momentum'] >= self.get_param('momentum_threshold')) &  # 动量确认
            (df['vol'] > df['vol'].mean() * self.get_param('volume_multiplier')) &  # 放量
            (abs(df['close'] - df['ma5']) / df['close'] >= self.get_param('ma_distance_threshold')) &  # 远离MA
            (df['atr_pct'] <= self.get_param('max_atr_ratio'))  # 波动率合适
        )
        
        # 卖出信号筛选
        sell_mask = (
            (df['low_breakout'] > 0) &  # 低点突破
            (df['momentum'] <= -self.get_param('momentum_threshold')) &  # 动量确认
            (df['vol'] > df['vol'].mean() * self.get_param('volume_multiplier')) &  # 放量
            (abs(df['close'] - df['ma5']) / df['close'] >= self.get_param('ma_distance_threshold')) &  # 远离MA
            (df['atr_pct'] <= self.get_param('max_atr_ratio'))  # 波动率合适
        )
        
        buy_candidates = df[buy_mask]
        sell_candidates = df[sell_mask]
        
        # 选择最佳候选 (按动量排序)
        if not buy_candidates.empty:
            # 按动量排序
            top_buy = buy_candidates.nlargest(3, 'momentum')
            for _, row in top_buy.iterrows():
                code = row['code']
                if code not in self._positions:
                    self._positions[code] = UltraShortPosition(
                        code=code,
                        side=OrderSide.BUY,
                        entry_price=row['close'],
                        entry_time=datetime.now(),
                        entry_bar_index=self._current_bar_index,
                        quantity=0,  # 由引擎计算
                        initial_momentum=row['momentum']
                    )
                    
                    signals.append(Signal(
                        code=code,
                        side=OrderSide.BUY,
                        weight=base_position_pct,
                        price=row['close'],
                        reason=f"超短线突破 动量={row['momentum']:.2%}"
                    ))
                    
                    if len(self._positions) >= max_concurrent:
                        break
        
        if not sell_candidates.empty:
            # 按动量排序 (取最小的，即负值最大的)
            top_sell = sell_candidates.nsmallest(3, 'momentum')
            for _, row in top_sell.iterrows():
                code = row['code']
                if code not in self._positions:
                    self._positions[code] = UltraShortPosition(
                        code=code,
                        side=OrderSide.SELL,
                        entry_price=row['close'],
                        entry_time=datetime.now(),
                        entry_bar_index=self._current_bar_index,
                        quantity=0,
                        initial_momentum=row['momentum']
                    )
                    
                    signals.append(Signal(
                        code=code,
                        side=OrderSide.SELL,
                        weight=base_position_pct,
                        price=row['close'],
                        reason=f"超短线突破 动量={row['momentum']:.2%}"
                    ))
                    
                    if len(self._positions) >= max_concurrent:
                        break
        
        return signals
    
    def _create_exit_signal(self, pos: UltraShortPosition, current_price: float, 
                            reason: str) -> Optional[Signal]:
        """创建离场信号"""
        return Signal(
            code=pos.code,
            side=OrderSide.BUY if pos.side == OrderSide.SELL else OrderSide.SELL,  # 反向信号平仓
            weight=0.0,
            price=current_price,
            reason=reason
        )
    
    def on_order_filled(self, order: 'Order') -> None:
        """订单成交回调 - 记录交易"""
        # 更新每日交易次数
        self._daily_trades += 1
        
        # 查找对应的持仓
        for code, pos in list(self._positions.items()):
            if code == order.code:
                # 如果是平仓订单，记录交易
                if order.side != pos.side:
                    pnl_pct = (order.price - pos.entry_price) / pos.entry_price
                    if pos.side == OrderSide.SELL:
                        pnl_pct = -pnl_pct
                    
                    # 记录交易
                    self._trade_records.append(TradeRecord(
                        timestamp=datetime.now(),
                        code=code,
                        side=pos.side,
                        entry_price=pos.entry_price,
                        exit_price=order.price,
                        pnl=0,  # 由引擎计算
                        pnl_pct=pnl_pct,
                        bars_held=pos.bars_held
                    ))
                    
                    # 更新统计
                    self._daily_pnl += pnl_pct * order.quantity * pos.entry_price
                    
                    if pnl_pct < 0:
                        self._consecutive_losses += 1
                        self._last_loss_time = datetime.now()
                    else:
                        self._consecutive_losses = 0
                    
                    # 检查是否需要暂停
                    if self._consecutive_losses >= self.get_param('max_consecutive_losses'):
                        self._is_paused = True
                        self._pause_end_time = datetime.now() + timedelta(
                            hours=self.get_param('pause_after_losses_hours')
                        )
                        self.logger.warning(
                            f"连续{self._consecutive_losses}次亏损，"
                            f"暂停交易{self.get_param('pause_after_losses_hours')}小时"
                        )
                break
    
    def on_day_end(self, context: StrategyContext) -> None:
        """日终回调 - 重置统计"""
        self._daily_trades = 0
        self._daily_pnl = 0.0
        self._daily_equity_start = context.total_equity
        
        # 清理已完成持仓
        self._positions.clear()
